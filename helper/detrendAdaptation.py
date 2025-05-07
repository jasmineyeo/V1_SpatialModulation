import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from matplotlib.gridspec import GridSpec
import seaborn as sns

class detrendAdaptation:
    """Class to detrend and adapt neural data for analysis."""
    
    @staticmethod      
    def detrend_time_component(spatial_activity, bin_centers, halfway_point=None):
        """
        Use regression to separate time (adaptation) and spatial components in neural activity.
        
        Parameters:
        -----------
        spatial_activity : numpy.ndarray
            Activity matrix (cells x trials x spatial_bins)
        bin_centers : numpy.ndarray
            Centers of spatial bins
        halfway_point : float, optional
            Halfway point of the corridor. If None, calculated from bin_centers.
            
        Returns:
        --------
        detrended_activity : numpy.ndarray
            Activity with time component removed (cells x trials x spatial_bins)
        time_coefficients : numpy.ndarray
            Time coefficient for each cell (cells,)
        spatial_coefficients : numpy.ndarray
            Spatial coefficient for each bin for each cell (cells x spatial_bins)
        """
        print("Performing regression-based detrending to remove adaptation effects...")
        
        n_cells, n_trials, n_bins = spatial_activity.shape
        
        # Set halfway point if not provided
        if halfway_point is None:
            halfway_point = np.median(bin_centers)
        
        # Prepare normalized time (0 to 1 for each spatial bin)
        normalized_time = np.linspace(0, 1, n_bins)
        
        # Initialize arrays for results
        detrended_activity = np.zeros_like(spatial_activity)
        time_coefficients = np.zeros(n_cells)
        spatial_coefficients = np.zeros((n_cells, n_bins))
        r_squared_values = np.zeros(n_cells)
        
        # Perform regression for each cell
        for cell in range(n_cells):
            # Average across trials for more stable estimation
            cell_activity = np.mean(spatial_activity[cell], axis=0)
            
            # Design matrix: intercept, time component
            X = np.ones((n_bins, 2)) ## Intercept and time component, column 1: all one (for the baseline activity), column 2: normalized time
            X[:, 1] = normalized_time
            
            # Fit linear regression
            beta, _, _, _ = np.linalg.lstsq(X, cell_activity, rcond=None)
            
            # Extract coefficients
            intercept = beta[0]
            time_coef = beta[1]
            time_coefficients[cell] = time_coef
            
            # Calculate predicted activity based on time component
            time_predicted = intercept + time_coef * normalized_time
            
            # Calculate R² value -- how much of the variation in neural activity can be explained by a simple linear time trend
            # r2 close to 1 = a large portion of activity variation is due to time-dependent adaptation
            # r2 close to 0 =  Time adaptation explains little of the activity pattern, suggesting spatial coding might be more important
            ss_total = np.sum((cell_activity - np.mean(cell_activity))**2)
            ss_residual = np.sum((cell_activity - time_predicted)**2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
            r_squared_values[cell] = r_squared
            
            # Store spatial component (residuals)
            spatial_component = cell_activity - time_predicted + np.mean(cell_activity)  # Add mean back for baseline
            spatial_coefficients[cell] = spatial_component
            
            # Apply detrending to all trials for this cell
            for trial in range(n_trials):
                trial_data = spatial_activity[cell, trial, :]
                # Remove time trend but keep overall activity level
                trial_detrended = trial_data - (time_coef * normalized_time) 
                detrended_activity[cell, trial, :] = trial_detrended
        
        # Create visualization of detrending for a few example cells
        detrendAdaptation.visualize_detrending_examples(spatial_activity, detrended_activity, 
                                    normalized_time, time_coefficients, 
                                    spatial_coefficients, r_squared_values, 
                                    bin_centers, n_examples=4)
        
        # Summarize time coefficients
        neg_time_coef = np.sum(time_coefficients < 0) / n_cells * 100
        print(f"Time component analysis:")
        print(f"  Cells with negative time coefficient (adaptation): {neg_time_coef:.1f}%")
        print(f"  Mean time coefficient: {np.mean(time_coefficients):.3f}")
        print(f"  Mean R² value: {np.mean(r_squared_values):.3f}")
        
        # Calculate average adaptation magnitude
        half_idx = n_bins // 2
        avg_adaptation_ratio = np.mean(spatial_activity[:, :, :half_idx]) / np.mean(spatial_activity[:, :, half_idx:])
        avg_detrended_ratio = np.mean(detrended_activity[:, :, :half_idx]) / np.mean(detrended_activity[:, :, half_idx:])
        
        print(f"  Original first/second half ratio: {avg_adaptation_ratio:.2f}")
        print(f"  Detrended first/second half ratio: {avg_detrended_ratio:.2f}")
        print(f"  Adaptation effect reduced by {(1 - avg_detrended_ratio/avg_adaptation_ratio)*100:.1f}%")
        
        return detrended_activity, time_coefficients, spatial_coefficients, r_squared_values

    @staticmethod
    def visualize_detrending_examples(spatial_activity, detrended_activity, 
                                    normalized_time, time_coefficients, 
                                    spatial_coefficients, r_squared_values, 
                                    bin_centers, n_examples=4):
        """
        Visualize examples of the detrending process.
        """
        # Choose cells with various adaptation strengths
        # Find cells with strong, medium, weak, and reverse adaptation
        sorted_indices = np.argsort(time_coefficients)
        n_cells = len(sorted_indices)
        
        # Get indices for example cells with different adaptation strengths
        strong_negative_idx = sorted_indices[int(0.05 * n_cells)]  # 5th percentile
        medium_negative_idx = sorted_indices[int(0.25 * n_cells)]  # 25th percentile
        weak_idx = sorted_indices[int(0.5 * n_cells)]             # 50th percentaile, median
        positive_idx = sorted_indices[int(0.95 * n_cells)]        # 95th percentile
        
        example_indices = [strong_negative_idx, medium_negative_idx, weak_idx, positive_idx]
        example_titles = ['Strong Adaptation', 'Medium Adaptation', 
                        'Weak Adaptation', 'Reverse Adaptation']
        
        # Create figure
        fig, axes = plt.subplots(n_examples, 3, figsize=(15, 4*n_examples))
        
        for i, (cell_idx, title) in enumerate(zip(example_indices, example_titles)):
            # Original activity (averaged across trials)
            orig_activity = np.mean(spatial_activity[cell_idx], axis=0)
            detrended = np.mean(detrended_activity[cell_idx], axis=0)
            
            # Create predicted time component
            time_coef = time_coefficients[cell_idx]
            intercept = np.mean(orig_activity) - time_coef * np.mean(normalized_time)
            time_predicted = intercept + time_coef * normalized_time
            
            # Plot original activity and time component
            ax = axes[i, 0]
            ax.plot(bin_centers, orig_activity, 'b-', label='Original activity')
            ax.plot(bin_centers, time_predicted, 'r--', label='Time component')
            ax.set_title(f'{title} (R²={r_squared_values[cell_idx]:.2f})')
            ax.set_xlabel('Position (cm)')
            ax.set_ylabel('Activity')
            ax.legend()
            
            # Plot detrended activity
            ax = axes[i, 1]
            ax.plot(bin_centers, detrended, 'g-')
            ax.set_title('Detrended Activity')
            ax.set_xlabel('Position (cm)')
            ax.set_ylabel('Activity')
            
            # Plot all trials before and after
            ax = axes[i, 2]
            for trial in range(min(5, spatial_activity.shape[1])):  # Plot up to 5 trials
                alpha = 0.2
                ax.plot(bin_centers, spatial_activity[cell_idx, trial, :], 'b-', alpha=alpha)
                ax.plot(bin_centers, detrended_activity[cell_idx, trial, :], 'g-', alpha=alpha)
            
            # Add averages with higher opacity
            ax.plot(bin_centers, np.mean(spatial_activity[cell_idx], axis=0), 'b-', 
                linewidth=2, label='Original')
            ax.plot(bin_centers, np.mean(detrended_activity[cell_idx], axis=0), 'g-', 
                linewidth=2, label='Detrended')
            
            ax.set_title('Before vs After (Multiple Trials)')
            ax.set_xlabel('Position (cm)')
            ax.set_ylabel('Activity')
            ax.legend()
        
        plt.tight_layout()
        plt.show()

    @staticmethod
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
    
    @staticmethod
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
            popt, _ = curve_fit(detrendAdaptation.double_gaussian, x_data, y_data, p0=p0, bounds=bounds, maxfev=2000)
            
            # Generate the fitted curve across all bin positions
            fit_curve = detrendAdaptation.double_gaussian(bin_centers, *popt)
            
            # Extract peak position and value from fitted parameters
            peak_position = popt[1]  # mu parameter
            peak_response = popt[0]  # A parameter
            
            return popt, fit_curve, peak_position, peak_response, True
            
        except (RuntimeError, ValueError) as e:
            # print(f"Fitting failed at index {initial_peak_idx}: {str(e)}")
            # Fall back to the original peak if fitting fails
            return None, None, bin_centers[initial_peak_idx], profile[initial_peak_idx], False

    @staticmethod
    def calculate_detrended_SMI(detrended_activity, bin_centers, reliable_cells, segment_distance=52, exclude_boundary_cm=4):
        """
        Calculate the Spatial Modulation Index (SMI) using cross-validation approach with Gaussian fitting:
        - Odd trials to find preferred position
        - Even trials to measure responses
        - SMI = (Rp - Rn) / (Rp + Rn)
        
        Where Rp = response at preferred position, Rn = response at non-preferred position.
        """
        n_cells, n_trials, n_bins = detrended_activity.shape
        
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
        odd_profiles = np.mean(detrended_activity[:, odd_indices, :], axis=1)
        even_profiles = np.mean(detrended_activity[:, even_indices, :], axis=1)
        
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
            popt_odd, fit_curve_odd, preferred_position_fitted_odd, _, fit_success_odd = detrendAdaptation.fit_response_profile(
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
            popt_even_pref, fit_curve_even_pref, preferred_position_fitted_even, peak_response_even, fit_success_even = detrendAdaptation.fit_response_profile(
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
            popt_even_nonpref, fit_curve_even_nonpref, non_preferred_position_fitted, non_preferred_resp_fitted, fit_success_nonpref = detrendAdaptation.fit_response_profile(
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

    @staticmethod    
    def calculate_detrended_SMI_BBBB(spatial_activity, bin_centers, reliable_cells, segment_distances=[20, 40], exclude_boundary_cm=0):
        """
        Calculate the Spatial Modulation Index (SMI) using cross-validation approach with Gaussian fitting:
        - Odd trials to find preferred position
        - Even trials to measure responses at preferred and non-preferred positions
        - SMI = (Rp - Rn) / (Rp + Rn)
        
        With three landmarks, creating multiple possible non-preferred positions.
        Where Rp = response at preferred position, Rn = response at non-preferred position.
        """
        n_cells, n_trials, n_bins = spatial_activity.shape
        
        # RANDOM HALVES instead of odd/even
        odd_indices = np.random.choice(n_trials, n_trials // 2, replace=False)
        even_indices = np.setdiff1d(np.arange(n_trials), odd_indices)
        
        # Calculate corridor boundaries
        min_pos = np.min(bin_centers)
        max_pos = np.max(bin_centers)
        corridor_length = max_pos - min_pos
        
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
        
        # Track which landmark segment the preferred position falls near
        landmark_segment = np.zeros(n_cells, dtype=int)  # 0=first, 1=second, 2=third
        
        # Store all potential non-preferred positions and responses for visualization
        all_potential_nonpref = np.zeros((n_cells, len(segment_distances) * 2, 3))  # [cell, option, [position, response, valid]]

        # Count various rejection reasons
        outside_boundary_count = 0
        nonpref_outside_range_count = 0
        zero_response_count = 0
        fitting_failed_count = 0
        valid_count = 0
        
        # Define the landmarks (assuming they're evenly spaced)
        corridor_third = corridor_length / 3
        landmark_positions = [
            min_pos + corridor_third,         # First landmark
            min_pos + 2 * corridor_third,     # Second landmark
            max_pos                           # Third landmark
        ]
        
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
            popt_odd, fit_curve_odd, preferred_position_fitted_odd, _, fit_success_odd = detrendAdaptation.fit_response_profile(
                bin_centers, odd_profiles[cell], preferred_idx_odd, window_size=5
            )
            
            # Find the closest bin to the fitted preferred position
            preferred_idx_odd_fitted = np.argmin(np.abs(bin_centers - preferred_position_fitted_odd))
            
            # Now find the corresponding peak in even trials within a window of the refined odd peak
            start_idx_pref = max(0, preferred_idx_odd_fitted - 1)
            end_idx_pref = min(n_bins, preferred_idx_odd_fitted + 1)
            window_profile_pref = even_profiles[cell, start_idx_pref:end_idx_pref]
            window_max_idx_pref = np.argmax(window_profile_pref)
            preferred_idx_even_initial = start_idx_pref + window_max_idx_pref
            
            # Fit a double Gaussian to the even trials around this peak for the final preferred position
            popt_even_pref, fit_curve_even_pref, preferred_position_fitted_even, peak_response_even, fit_success_even = detrendAdaptation.fit_response_profile(
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

            # Determine which landmark segment the preferred position is closest to
            distances_to_landmarks = [abs(preferred_position_even - pos) for pos in landmark_positions]
            closest_landmark = np.argmin(distances_to_landmarks)
            landmark_segment[cell] = closest_landmark
            
            # Calculate potential non-preferred positions based on which landmark segment we're in
            non_preferred_positions_list = []
            
            if closest_landmark == 0:  # Near first landmark
                # Non-preferred can be +20 and +40 from preferred
                for dist in segment_distances:
                    non_preferred_positions_list.append(preferred_position_even + dist)
            elif closest_landmark == 1:  # Near second landmark
                # Non-preferred can be -20 and +20 from preferred
                non_preferred_positions_list.append(preferred_position_even - segment_distances[0])
                non_preferred_positions_list.append(preferred_position_even + segment_distances[0])
            else:  # Near third landmark
                # Non-preferred can be -20 and -40 from preferred
                for dist in segment_distances:
                    non_preferred_positions_list.append(preferred_position_even - dist)
            
            # Check if any non-preferred position is outside corridor bounds
            all_outside = True
            for pos in non_preferred_positions_list:
                if min_pos <= pos <= max_pos:
                    all_outside = False
                    break
                    
            if all_outside:
                nonpref_outside_range_count += 1
                valid_cells[cell] = False
                continue
            
            # Process each potential non-preferred position
            non_preferred_responses = []
            valid_non_preferred_positions = []
            fitted_non_preferred_curves = []
            non_preferred_fit_success = []
            
            # Initialize array to store all potential non-preferred positions for this cell
            cell_potential_nonpref = np.zeros((len(non_preferred_positions_list), 3))
            
            for i, non_pref_pos in enumerate(non_preferred_positions_list):
                # Set default values for invalid positions
                cell_potential_nonpref[i, 0] = non_pref_pos  # Position
                cell_potential_nonpref[i, 1] = -1  # Response (invalid)
                cell_potential_nonpref[i, 2] = 0   # Valid flag (0=invalid)
                
                # Skip if outside corridor bounds
                if non_pref_pos < min_pos or non_pref_pos > max_pos:
                    continue
                    
                # Find the closest bin to this non-preferred position
                non_preferred_idx_approx = np.argmin(np.abs(bin_centers - non_pref_pos))
                
                # Get exact response at this position (no window, just the exact position)
                exact_response = even_profiles[cell, non_preferred_idx_approx]
                
                # Fit a double Gaussian to the non-preferred position response
                # Use a window to allow finding the precise location
                start_idx_nonpref = max(0, non_preferred_idx_approx - 5)
                end_idx_nonpref = min(n_bins, non_preferred_idx_approx + 5)
                
                if start_idx_nonpref >= end_idx_nonpref:
                    continue  # Skip if window is invalid
                    
                window_profile_nonpref = even_profiles[cell, start_idx_nonpref:end_idx_nonpref]
                window_peak_idx = np.argmax(window_profile_nonpref) + start_idx_nonpref
                
                popt_even_nonpref, fit_curve, non_preferred_position_fitted, non_preferred_resp_fitted, fit_success_nonpref = detrendAdaptation.fit_response_profile(
                    bin_centers, even_profiles[cell], window_peak_idx, window_size=5
                )
                
                # If fitting succeeded, use the fitted values
                if fit_success_nonpref:
                    # Calculate the distance from the estimated position to the fitted position
                    distance_from_target = abs(non_pref_pos - non_preferred_position_fitted)
                    
                    # Only use the fitted position if it's close to our target position
                    if distance_from_target <= 5:  # Within 5 units of target
                        non_preferred_pos_final = non_preferred_position_fitted
                        non_preferred_resp = non_preferred_resp_fitted
                        fitted_curve_final = fit_curve
                        fit_success = True
                    else:
                        # If fitted position is too far, use the exact response at the target
                        non_preferred_pos_final = non_pref_pos
                        non_preferred_resp = exact_response
                        fitted_curve_final = np.zeros_like(bin_centers)
                        fit_success = False
                else:
                    # If fitting failed, use the exact response at the target
                    non_preferred_pos_final = non_pref_pos
                    non_preferred_resp = exact_response
                    fitted_curve_final = np.zeros_like(bin_centers)
                    fit_success = False
                
                # Store this position, its response, and its fitted curve
                valid_non_preferred_positions.append(non_preferred_pos_final)
                non_preferred_responses.append(non_preferred_resp)
                fitted_non_preferred_curves.append(fitted_curve_final)
                non_preferred_fit_success.append(fit_success)
                
                # Store for visualization
                cell_potential_nonpref[i, 0] = non_preferred_pos_final
                cell_potential_nonpref[i, 1] = non_preferred_resp
                cell_potential_nonpref[i, 2] = 1  # Valid flag
            
            # Store all potential non-preferred positions for this cell
            if cell < all_potential_nonpref.shape[0]:
                all_potential_nonpref[cell, :len(cell_potential_nonpref)] = cell_potential_nonpref
            
            # If no valid non-preferred positions, skip this cell
            if len(non_preferred_responses) == 0:
                nonpref_outside_range_count += 1
                valid_cells[cell] = False
                continue
            
            # Find the non-preferred position with the SMALLEST response (for maximal contrast)
            min_resp_idx = np.argmin(non_preferred_responses)
            Rn = non_preferred_responses[min_resp_idx]
            non_preferred_position_even = valid_non_preferred_positions[min_resp_idx]
            
            # Store the fitted curve for the chosen non-preferred position
            if non_preferred_fit_success[min_resp_idx]:
                non_preferred_fitted_curves[cell] = fitted_non_preferred_curves[min_resp_idx]
                fitting_success[cell, 1] = True
            else:
                fitting_success[cell, 1] = False
            
            # Calculate SMI - only allow positive SMI values (Rp must be > Rn)
            if Rp + Rn > 0:  # Avoid division by zero
                    # SMI = (Rp - Rn) / (Rp + Rn)
                    # valid_count += 1

                    if Rp - Rn > 0:
                        SMI = (Rp - Rn) / (Rp + Rn)
                        valid_count += 1
                    else:
                        SMI = 0
                        zero_response_count += 1
                        valid_cells[cell] = False
                        continue
                    
            else:
                # Skip cells where preferred response is not greater than non-preferred
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
        print(f"size of valid_cells: {valid_cells.shape}")
        print(f"size of reliable_cells: {reliable_cells.shape}")
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
        print(f"  Rejected - zero response sum or Rp ≤ Rn: {zero_response_count} ({zero_response_count/n_cells*100:.1f}%)")
        print(f"  Fitting failed (but used raw peak instead): {fitting_failed_count} ({fitting_failed_count/n_cells*100:.1f}%)")
        
        # Calculate segment statistics
        if np.sum(valid_cells) > 0:
            segment_counts = np.zeros(3, dtype=int)
            for i in range(3):
                segment_counts[i] = np.sum(landmark_segment[valid_cells] == i)
            
            print(f"\nPreferred position by landmark segment:")
            for i in range(3):
                print(f"  Landmark {i+1}: {segment_counts[i]} cells ({segment_counts[i]/np.sum(valid_cells)*100:.1f}%)")
            
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
            'landmark_segment': landmark_segment,
            'min_allowed': min_allowed,
            'max_allowed': max_allowed,
            'valid_cells': valid_cells,
            'reliable_valid_cells': reliable_valid_cells if reliable_cells is not None else None,
            'all_potential_nonpref': all_potential_nonpref,
            'parameters': {
                'segment_distances': segment_distances,
                'exclude_boundary_cm': exclude_boundary_cm,
                'n_cells': n_cells,
                'n_trials': n_trials,
                'n_bins': n_bins,
                'corridor_length': corridor_length,
                'min_pos': min_pos,
                'max_pos': max_pos,
                'landmark_positions': landmark_positions
            }
        }
        
        return results

    @staticmethod
    def analyze_layer_specific_detrended_SMI(detrended_smi_results, layer_cells, reliable_cells):
        """
        Analyze layer-specific SMI results after detrending.
        
        Parameters:
        -----------
        detrended_smi_results : dict
            Dictionary with SMI calculation results on detrended data
        layer_cells : dict
            Dictionary with indices of cells in each layer
        reliable_cells : numpy.ndarray
            Boolean array indicating reliable cells
            
        Returns:
        --------
        layer_results : dict
            Dictionary with layer-specific detrended SMI results
        """
        print("\nAnalyzing layer-specific detrended SMI...")
        
        # Get SMI values and valid cells
        SMI_values = detrended_smi_results['SMI']
        reliable_valid_cells = detrended_smi_results['reliable_valid_cells']
        
        # Find indices of reliable and valid cells
        reliable_valid_indices = np.where(reliable_valid_cells)[0]
        
        print(f"Total cells: {len(SMI_values)} and reliable & valid cells: {np.sum(reliable_valid_cells)}")
        
        # Initialize results dictionary for each layer
        layer_results = {}
        reliable_layer_cells_num = {}
        
        # Track how many cells we've classified into layers
        classified_cell_count = 0
        
        # Process each layer
        for layer_name, layer_cell_indices in layer_cells.items():
            # Find cells in this layer that are both reliable and valid
            layer_reliable_valid_cells = np.intersect1d(reliable_valid_indices, layer_cell_indices)
            classified_cell_count += len(layer_reliable_valid_cells)
            
            # Count reliable cells in this layer (for reference)
            reliable_layer_cells = np.intersect1d(np.where(reliable_cells)[0], layer_cell_indices)
            reliable_layer_cells_num[layer_name] = len(reliable_layer_cells)
            
            if len(layer_reliable_valid_cells) == 0:
                print(f"{layer_name}: No reliable valid cells found")
                layer_results[layer_name] = None
                continue
            
            # Extract SMI values and other metrics directly for these cells
            valid_smi = SMI_values[layer_reliable_valid_cells]
            
            # Check for NaN values
            nan_mask = np.isnan(valid_smi)
            if np.any(nan_mask):
                print(f"Warning: Found {np.sum(nan_mask)} NaN SMI values in {layer_name}")
                valid_smi = valid_smi[~nan_mask]
                layer_reliable_valid_cells = layer_reliable_valid_cells[~nan_mask]
            
            # Extract other metrics if available in smi_results
            valid_pref_pos = detrended_smi_results['preferred_positions'][layer_reliable_valid_cells] if 'preferred_positions' in detrended_smi_results else None
            valid_nonpref_pos = detrended_smi_results['non_preferred_positions'][layer_reliable_valid_cells] if 'non_preferred_positions' in detrended_smi_results else None
            valid_rp = detrended_smi_results['Rp'][layer_reliable_valid_cells] if 'Rp' in detrended_smi_results else None
            valid_rn = detrended_smi_results['Rn'][layer_reliable_valid_cells] if 'Rn' in detrended_smi_results else None
            
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
                print(f"Wilcoxon test failed for layer {layer_name}. This can happen if all values are identical.")
            
            # Clip SMI values to [-1, 1] range
            valid_smi[valid_smi < -1] = -1 
            valid_smi[valid_smi > 1] = 1
            
            # Store results
            layer_results[layer_name] = {
                'reliable_valid_cells': layer_reliable_valid_cells,
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
            print(f"{layer_name}: {len(layer_reliable_valid_cells)}/{reliable_layer_cells_num[layer_name]} valid cells")
            print(f"  Mean SMI = {mean_smi:.3f} ± {sem_smi:.3f} (SEM)")
            print(f"  Median SMI = {median_smi:.3f}")
            if not np.isnan(p_value):
                print(f"  Wilcoxon test against 1: W = {t_stat:.3f}, p = {p_value:.5f}")
            print()
        
        # Check if all reliable valid cells were assigned to a layer
        if classified_cell_count < len(reliable_valid_indices):
            print(f"Warning: {len(reliable_valid_indices) - classified_cell_count} reliable valid cells were not assigned to any layer!")
        elif classified_cell_count > len(reliable_valid_indices):
            print(f"Warning: {classified_cell_count - len(reliable_valid_indices)} cells were counted multiple times (assigned to multiple layers)!")
        else:
            print(f"All {len(reliable_valid_indices)} reliable valid cells were successfully assigned to layers.")
        
        # print(layer_results.items())
        detrendAdaptation.visualize_layer_specific_smi(layer_results, reliable_valid_cells, layer_cells)

        return layer_results

    @staticmethod
    def visualize_layer_specific_smi(layer_results, reliable_cells, layer_cells):
        """
        Create visualization for layer-specific SMI results.
        """
        # Setup figure
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=fig)
        
        # Plot 1: Box plot of SMI by layer
        ax1 = fig.add_subplot(gs[0, 0])
        
        boxplot_data = []
        layer_names = []
        mean_values = []
        sem_values = []
    
        for layer_name, results in layer_results.items():
            if results is not None and len(results['SMI']) > 0:
                boxplot_data.append(results['SMI'])
                layer_names.append(f"{layer_name}\n(n={results['reliable_valid_cells'].size})")
                # Calculate statistics directly from the same data
                mean_values.append(np.mean(results['SMI']))
                sem_values.append(stats.sem(results['SMI']))
        
        
        # Print values to verify
        print("Values being used in both plots:")
        for i, layer in enumerate(layer_names):
            print(f"  {layer}: mean={mean_values[i]:.3f}, sem={sem_values[i]:.3f}")
    
        boxplot = ax1.boxplot(boxplot_data, patch_artist=True)
        
        # Add colors to boxes
        colors = ['lightblue', 'lightgreen', 'salmon', 'plum']
        for patch, color in zip(boxplot['boxes'], colors[:len(boxplot_data)]):
            patch.set_facecolor(color)
        
        ax1.set_xticklabels(layer_names)
        ax1.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax1.set_title('Detrended SMI by Layer')
        ax1.set_ylabel('Spatial Modulation Index')
        
        # Plot 2: Bar plot with error bars
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Use the same calculated values for the bar chart
        bars = ax2.bar(range(len(mean_values)), mean_values, yerr=sem_values, 
                    capsize=10, color=colors[:len(mean_values)])
        
        ax2.set_xticks(range(len(mean_values)))
        ax2.set_xticklabels(layer_names)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax2.set_title('Mean Detrended SMI by Layer')
        ax2.set_ylabel('Mean SMI ± SEM')
            
        # Plot 3: Histogram of SMI distribution by layer
        ax3 = fig.add_subplot(gs[0, 2])
        
        for i, ((layer_name, results), color) in enumerate(zip(layer_results.items(), colors)):
            if results is not None and len(results['SMI']) > 0:
                # Create histogram
                ax3.hist(results['SMI'], bins=15, alpha=0.6, color=color, label=layer_name)
        
        ax3.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax3.set_title('Distribution of Detrended SMI')
        ax3.set_xlabel('SMI Value')
        ax3.set_ylabel('Count')
        ax3.legend()
        
        # # Plot 4: Comparison with original SMI (assuming we have it)
        # ax4 = fig.add_subplot(gs[1, 0:])
        
        # # Prepare data for layer-specific ANOVA
        # if len(layer_results) > 1:
        #     print("\nStatistical comparison between layers:")
        #     layer_names_stats = []
            
        #     # Perform pairwise t-tests
        #     layers = list(layer_results.keys())
            
        #     for i, layer1 in enumerate(layers):
        #         if layer_results[layer1] is None:
        #             continue
                    
        #         for j, layer2 in enumerate(layers[i+1:], i+1):
        #             if layer_results[layer2] is None:
        #                 continue
                        
        #             smi1 = layer_results[layer1]['SMI']
        #             smi2 = layer_results[layer2]['SMI']
                    
        #             # Perform t-test
        #             t_stat, p_value = stats.ttest_ind(smi1, smi2, equal_var=False)
                    
        #             print(f"{layer1} vs {layer2}: t = {t_stat:.5f}, p = {p_value:.5f}")
                    
        #             if p_value < 0.05:
        #                 print(f"  Significant difference in SMI between {layer1} and {layer2}")
        #                 if np.mean(smi1) > np.mean(smi2):
        #                     print(f"  {layer1} shows stronger spatial modulation than {layer2}")
        #                 else:
        #                     print(f"  {layer2} shows stronger spatial modulation than {layer1}")
        #             else:
        #                 print(f"  No significant difference in SMI between {layer1} and {layer2}")
                
        #         layer_names_stats.append(layer1)
        
        #     # ANOVA if more than 2 groups
        #     if len(layer_names_stats) > 2:
        #         anova_groups = [layer_results[layer]['SMI'] for layer in layer_names_stats 
        #                     if layer_results[layer] is not None]
                
        #         if all(len(group) > 0 for group in anova_groups):
        #             f_stat, p_anova = stats.f_oneway(*anova_groups)
        #             print(f"\nANOVA across all layers: F = {f_stat:.3f}, p = {p_anova:.5f}")
        #             if p_anova < 0.05:
        #                 print("  Significant differences exist between layers")
        #             else:
        #                 print("  No significant differences between layers")
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def run_adaptation_corrected_smi_analysis(spatial_activity, bin_centers, reliable_cells, layer_cells, segment_distance=52, exclude_boundary_cm=4):
        """
        Run complete analysis of SMI after correcting for adaptation effects.
        
        Parameters:
        -----------
        spatial_activity : numpy.ndarray
            Activity matrix (cells x trials x spatial_bins)
        bin_centers : numpy.ndarray
            Centers of spatial bins
        reliable_cells : numpy.ndarray
            Boolean array indicating reliable cells
        layer_cells : dict
            Dictionary with indices of cells in each layer
        segment_distance : float
            Distance between visually identical positions
        exclude_boundary_cm : float
            Distance from corridor boundaries to exclude
        """
        # Step 1: Remove time component (adaptation) from neural activity
        detrended_activity, time_coefficients, spatial_coefficients, r_squared_values = detrendAdaptation.detrend_time_component(
            spatial_activity, bin_centers)
        
        # Step 2: Calculate SMI on detrended activity
        detrended_smi_results = detrendAdaptation.calculate_detrended_SMI(
            detrended_activity, bin_centers, reliable_cells, segment_distance=40, exclude_boundary_cm=10)
            # detrended_activity, bin_centers, reliable_cells, segment_distance=40, exclude_boundary_cm=10)

        # Step 3: Analyze layer-specific detrended SMI
        layer_results = detrendAdaptation.analyze_layer_specific_detrended_SMI(
            detrended_smi_results, layer_cells, reliable_cells)
        
        # # Step 4: Compare time coefficients (adaptation strength) across layers
        # detrendAdaptation.analyze_layer_specific_adaptation(
        #     time_coefficients, r_squared_values, layer_cells, reliable_cells)
        
        return detrended_activity, detrended_smi_results, layer_results

    @staticmethod
    def run_adaptation_corrected_smi_analysis_BBBB(spatial_activity, bin_centers, reliable_cells, layer_cells, segment_distance=52, exclude_boundary_cm=4):
        """
        Run complete analysis of SMI after correcting for adaptation effects.
        
        Parameters:
        -----------
        spatial_activity : numpy.ndarray
            Activity matrix (cells x trials x spatial_bins)
        bin_centers : numpy.ndarray
            Centers of spatial bins
        reliable_cells : numpy.ndarray
            Boolean array indicating reliable cells
        layer_cells : dict
            Dictionary with indices of cells in each layer
        segment_distance : float
            Distance between visually identical positions
        exclude_boundary_cm : float
            Distance from corridor boundaries to exclude
        """
        # Step 1: Remove time component (adaptation) from neural activity
        detrended_activity, time_coefficients, spatial_coefficients, r_squared_values = detrendAdaptation.detrend_time_component(
            spatial_activity, bin_centers)
        
        print(segment_distance, exclude_boundary_cm)
        # Step 2: Calculate SMI on detrended activity
        detrended_smi_results = detrendAdaptation.calculate_detrended_SMI_BBBB(
            detrended_activity, bin_centers, reliable_cells, segment_distances=segment_distance, exclude_boundary_cm = exclude_boundary_cm)
            # detrended_activity, bin_centers, reliable_cells, segment_distance=40, exclude_boundary_cm=10)

        # Step 3: Analyze layer-specific detrended SMI
        layer_results = detrendAdaptation.analyze_layer_specific_detrended_SMI(
            detrended_smi_results, layer_cells, reliable_cells)
        
        # # Step 4: Compare time coefficients (adaptation strength) across layers
        # detrendAdaptation.analyze_layer_specific_adaptation(
        #     time_coefficients, r_squared_values, layer_cells, reliable_cells)
        
        return detrended_activity, detrended_smi_results, layer_results
    
    @staticmethod
    def analyze_layer_specific_adaptation(time_coefficients, r_squared_values, layer_cells, reliable_cells):
        """
        Analyze adaptation strength (time coefficients) by layer.
        
        Parameters:
        -----------
        time_coefficients : numpy.ndarray
            Time coefficient for each cell
        r_squared_values : numpy.ndarray
            R² value for each cell's adaptation model
        layer_cells : dict
            Dictionary with indices of cells in each layer
        reliable_cells : numpy.ndarray
            Boolean array indicating reliable cells
        """
        print("\nAnalyzing adaptation strength by layer...")
        
        # Setup figure
        plt.figure(figsize=(15, 10))
        
        # Collect layer-specific time coefficients (adaptation strength)
        layer_time_coeffs = {}
        layer_r_squared = {}
        
        for layer_name, layer_cell_indices in layer_cells.items():
            # Consider only reliable cells
            reliable_layer_cells = np.intersect1d(np.where(reliable_cells)[0], layer_cell_indices)
            
            if len(reliable_layer_cells) > 0:
                # Extract time coefficients for this layer
                layer_time_coeffs[layer_name] = time_coefficients[reliable_layer_cells]
                layer_r_squared[layer_name] = r_squared_values[reliable_layer_cells]
                
                # Print statistics
                mean_coeff = np.mean(layer_time_coeffs[layer_name])
                median_coeff = np.median(layer_time_coeffs[layer_name])
                neg_percent = np.sum(layer_time_coeffs[layer_name] < 0) / len(layer_time_coeffs[layer_name]) * 100
                
                print(f"{layer_name}:")
                print(f"  Mean time coefficient: {mean_coeff:.3f}")
                print(f"  Median time coefficient: {median_coeff:.3f}")
                print(f"  Percent with negative coefficient (adaptation): {neg_percent:.1f}%")
                print(f"  Mean R² value: {np.mean(layer_r_squared[layer_name]):.3f}")
        
        # Plot 1: Box plot of time coefficients by layer
        plt.subplot(2, 2, 1)
        
        boxplot_data = []
        layer_names = []
        
        for layer_name, coeffs in layer_time_coeffs.items():
            boxplot_data.append(coeffs)
            layer_names.append(f"{layer_name}\n(n={len(coeffs)})")
        
        plt.boxplot(boxplot_data)
        plt.axhline(0, color='r', linestyle='--', alpha=0.5)
        plt.xticks(range(1, len(layer_names) + 1), layer_names)
        plt.title('Adaptation Strength by Layer')
        plt.ylabel('Time Coefficient')
        
        # Plot 2: Histogram of time coefficients by layer
        plt.subplot(2, 2, 2)
        
        colors = ['blue', 'green', 'red', 'purple']
        
        for i, (layer_name, coeffs) in enumerate(layer_time_coeffs.items()):
            plt.hist(coeffs, bins=15, alpha=0.5, label=layer_name, color=colors[i % len(colors)])
        
        plt.axvline(0, color='k', linestyle='--', alpha=0.5)
        plt.title('Distribution of Adaptation Strength')
        plt.xlabel('Time Coefficient')
        plt.ylabel('Count')
        plt.legend()
        
        # Plot 3: R² values by layer
        plt.subplot(2, 2, 3)
        
        boxplot_data = []
        layer_names = []
        
        for layer_name, r_squared in layer_r_squared.items():
            boxplot_data.append(r_squared)
            layer_names.append(f"{layer_name}\n(n={len(r_squared)})")
        
        plt.boxplot(boxplot_data)
        plt.xticks(range(1, len(layer_names) + 1), layer_names)
        plt.title('Model Fit Quality by Layer')
        plt.ylabel('R² Value')
        
        # Plot 4: Scatter plot of time coefficient vs R²
        plt.subplot(2, 2, 4)
        
        for i, layer_name in enumerate(layer_time_coeffs.keys()):
            plt.scatter(layer_time_coeffs[layer_name], layer_r_squared[layer_name], 
                    alpha=0.6, label=layer_name, color=colors[i % len(colors)])
        
        plt.axvline(0, color='k', linestyle='--', alpha=0.5)
        plt.title('Adaptation Strength vs Model Fit Quality')
        plt.xlabel('Time Coefficient')
        plt.ylabel('R² Value')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Statistical comparisons between layers
        if len(layer_time_coeffs) > 1:
            print("\nStatistical comparison of adaptation strength between layers:")
            
            layer_keys = list(layer_time_coeffs.keys())
            
            for i, layer1 in enumerate(layer_keys):
                for j, layer2 in enumerate(layer_keys[i+1:], i+1):
                    coeffs1 = layer_time_coeffs[layer1]
                    coeffs2 = layer_time_coeffs[layer2]
                    
                    t_stat, p_value = stats.ttest_ind(coeffs1, coeffs2, equal_var=False)
                    print(f"  {layer1} vs {layer2}: t={t_stat:.3f}, p={p_value:.4f}")
                    
                    if p_value < 0.05:
                        stronger = layer1 if np.mean(coeffs1) < np.mean(coeffs2) else layer2
                        print(f"    {stronger} shows significantly stronger adaptation")

    @staticmethod
    def compare_original_vs_detrended_smi(original_smi, detrended_smi, layer_cells, reliable_cells):
        """
        Compare original SMI values with detrended SMI values.
        
        Parameters:
        -----------
        original_smi : dict
            Results from original SMI calculation
        detrended_smi : dict
            Results from detrended SMI calculation
        layer_cells : dict
            Dictionary with indices of cells in each layer
        reliable_cells : numpy.ndarray
            Boolean array indicating reliable cells
        """
        print("\nComparing original vs detrended SMI...")
        
        # Extract SMI values
        orig_smi = original_smi['SMI']
        detr_smi = detrended_smi['SMI']
        
        # Extract valid cells
        orig_valid = original_smi['valid_cells']
        detr_valid = detrended_smi['valid_cells']
        
        # Find cells valid in both analyses
        both_valid = np.logical_and(orig_valid, detr_valid)
        both_valid = np.logical_and(both_valid, reliable_cells)
        
        valid_indices = np.where(both_valid)[0]
        
        if len(valid_indices) == 0:
            print("No cells valid in both original and detrended analyses")
            return
        
        # Extract SMI values for valid cells
        orig_smi_valid = orig_smi[valid_indices]
        detr_smi_valid = detr_smi[valid_indices]
        
        # Calculate overall statistics
        mean_orig = np.mean(orig_smi_valid)
        mean_detr = np.mean(detr_smi_valid)
        
        print(f"Overall comparison (n={len(valid_indices)} cells):")
        print(f"  Mean original SMI: {mean_orig:.3f}")
        print(f"  Mean detrended SMI: {mean_detr:.3f}")
        print(f"  Change: {(mean_detr - mean_orig):.3f} ({(mean_detr - mean_orig)/mean_orig*100:.1f}%)")
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(orig_smi_valid, detr_smi_valid)
        print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            direction = "increased" if mean_detr > mean_orig else "decreased"
            print(f"  SMI significantly {direction} after detrending")
        else:
            print("  No significant change in SMI after detrending")
        
        # Calculate correlation between original and detrended SMI
        corr, p_corr = stats.pearsonr(orig_smi_valid, detr_smi_valid)
        print(f"  Correlation between original and detrended SMI: r={corr:.3f}, p={p_corr:.4f}")
        
        # Layer-specific comparison
        print("\nLayer-specific comparison:")
        
        for layer_name, layer_cell_indices in layer_cells.items():
            # Find cells in this layer that are valid in both analyses
            layer_valid = np.intersect1d(valid_indices, layer_cell_indices)
            
            if len(layer_valid) > 0:
                layer_orig_smi = orig_smi[layer_valid]
                layer_detr_smi = detr_smi[layer_valid]
                
                mean_layer_orig = np.mean(layer_orig_smi)
                mean_layer_detr = np.mean(layer_detr_smi)
                
                print(f"  {layer_name} (n={len(layer_valid)} cells):")
                print(f"    Mean original SMI: {mean_layer_orig:.3f}")
                print(f"    Mean detrended SMI: {mean_layer_detr:.3f}")
                print(f"    Change: {(mean_layer_detr - mean_layer_orig):.3f} ({(mean_layer_detr - mean_layer_orig)/mean_layer_orig*100:.1f}%)")
                
                # Paired t-test
                if len(layer_valid) > 1:
                    t_stat, p_value = stats.ttest_rel(layer_orig_smi, layer_detr_smi)
                    print(f"    Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
                    
                    if p_value < 0.05:
                        direction = "increased" if mean_layer_detr > mean_layer_orig else "decreased"
                        print(f"    SMI significantly {direction} after detrending in {layer_name}")
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Scatter of original vs detrended SMI
        plt.subplot(2, 2, 1)
        
        for i, (layer_name, layer_cell_indices) in enumerate(layer_cells.items()):
            layer_valid = np.intersect1d(valid_indices, layer_cell_indices)
            
            if len(layer_valid) > 0:
                plt.scatter(orig_smi[layer_valid], detr_smi[layer_valid], 
                        alpha=0.6, label=f"{layer_name} (n={len(layer_valid)})")
        
        # Add unity line
        max_val = max(np.max(orig_smi_valid), np.max(detr_smi_valid))
        min_val = min(np.min(orig_smi_valid), np.min(detr_smi_valid))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        plt.title(f'Original vs Detrended SMI (r={corr:.2f}, p={p_corr:.4f})')
        plt.xlabel('Original SMI')
        plt.ylabel('Detrended SMI')
        plt.legend()
        
        # Plot 2: Histogram of differences
        plt.subplot(2, 2, 2)
        
        differences = detr_smi_valid - orig_smi_valid
        plt.hist(differences, bins=20)
        plt.axvline(0, color='r', linestyle='--', alpha=0.5)
        plt.title('Distribution of SMI Changes After Detrending')
        plt.xlabel('Detrended SMI - Original SMI')
        plt.ylabel('Count')
        
        # Plot 3: Bar plot of mean SMI by layer
        plt.subplot(2, 2, 3)
        
        layer_means_orig = []
        layer_means_detr = []
        layer_names = []
        
        for layer_name, layer_cell_indices in layer_cells.items():
            layer_valid = np.intersect1d(valid_indices, layer_cell_indices)
            
            if len(layer_valid) > 0:
                layer_means_orig.append(np.mean(orig_smi[layer_valid]))
                layer_means_detr.append(np.mean(detr_smi[layer_valid]))
                layer_names.append(f"{layer_name}\n(n={len(layer_valid)})")
        
        x = np.arange(len(layer_names))
        width = 0.35
        
        plt.bar(x - width/2, layer_means_orig, width, label='Original')
        plt.bar(x + width/2, layer_means_detr, width, label='Detrended')
        
        plt.axhline(0, color='k', linestyle='--', alpha=0.5)
        plt.xticks(x, layer_names)
        plt.title('Mean SMI by Layer')
        plt.ylabel('Mean SMI')
        plt.legend()
        
        # Plot 4: Percent change by layer
        plt.subplot(2, 2, 4)
        
        pct_changes = []
        
        for orig, detr in zip(layer_means_orig, layer_means_detr):
            if orig != 0:
                pct_changes.append((detr - orig) / abs(orig) * 100)
            else:
                pct_changes.append(0)
        
        plt.bar(x, pct_changes)
        plt.axhline(0, color='k', linestyle='--', alpha=0.5)
        plt.xticks(x, layer_names)
        plt.title('Percent Change in SMI After Detrending')
        plt.ylabel('Percent Change')
        
        plt.tight_layout()
        plt.show()
