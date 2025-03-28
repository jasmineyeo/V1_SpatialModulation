import numpy as np
import matplotlib.pyplot as plt

def calculate_SMI(spatial_activity, bin_centers, segment_distance=55, exclude_boundary_cm=15):
    """
    Calculate the Spatial Modulation Index (SMI) using cross-validation approach:
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
    corridor_length = np.max(bin_centers) - min_pos
    
    # Calculate boundary positions in the original coordinate system
    min_allowed = min_pos + exclude_boundary_cm
    max_allowed = np.max(bin_centers) - exclude_boundary_cm
    
    # Compute response profiles for odd and even trials
    odd_profiles = np.mean(spatial_activity[:, odd_indices, :], axis=1)
    even_profiles = np.mean(spatial_activity[:, even_indices, :], axis=1)
    
    # Initialize arrays to store results
    SMI_values = np.zeros(n_cells)
    preferred_positions = np.zeros(n_cells)
    non_preferred_positions = np.zeros(n_cells)
    non_preferred_response = np.zeros(n_cells)
    Rp_values = np.zeros(n_cells)
    Rn_values = np.zeros(n_cells)
    valid_cells = np.zeros(n_cells, dtype=bool)
    
    for cell in range(n_cells):
        # Find the preferred position from odd trials
        preferred_idx = np.argmax(odd_profiles[cell])
        preferred_position = bin_centers[preferred_idx]
        
        # Check if the preferred position is within allowed boundaries
        if preferred_position < min_allowed or preferred_position > max_allowed:
            valid_cells[cell] = False
            continue
        
        # Calculate the non-preferred position (visually identical position)
        corridor_midpoint = corridor_length / 2
        if preferred_position < corridor_midpoint:
            # If in first segment, the non-preferred position is in second segment
            non_preferred_position = preferred_position + segment_distance
        else:
            # If in second segment, the non-preferred position is in first segment
            non_preferred_position = preferred_position - segment_distance
        
        # Check if non-preferred position is within corridor bounds
        if non_preferred_position < 0 or non_preferred_position > corridor_length:
            valid_cells[cell] = False
            continue
        
        # Find the closest bin to the non-preferred position
        non_preferred_idx = np.argmin(np.abs(bin_centers - non_preferred_position))
        # Ensure the slice is within bounds
        start_idx = max(0, non_preferred_idx - 5)
        end_idx = min(n_bins, non_preferred_idx + 5)
        non_preferred_resp = np.max(even_profiles[cell, start_idx:end_idx])
        # print("non preferred response is", non_preferred_resp)
         
        # Get responses at preferred and non-preferred positions from EVEN trials
        Rp = even_profiles[cell, preferred_idx]
        # Rn = non_preferred_resp
        Rn = even_profiles[cell, non_preferred_idx]
        
        # Calculate SMI
        if Rp + Rn > 0:  # Avoid division by zero
            SMI = (Rp - Rn) / (Rp + Rn)
        else:
            SMI = 0
            
        # Store results
        SMI_values[cell] = SMI
        preferred_positions[cell] = preferred_position
        non_preferred_positions[cell] = non_preferred_position
        non_preferred_response[cell] = non_preferred_resp
        Rp_values[cell] = Rp
        Rn_values[cell] = Rn
        valid_cells[cell] = True
    
    # Create result dictionary
    results = {
        'SMI': SMI_values,
        'preferred_positions': preferred_positions,
        'non_preferred_positions': non_preferred_positions,
        'non_preferred_response' : non_preferred_response,
        'Rp': Rp_values,
        'Rn': Rn_values,
        'valid_cells': valid_cells,
        'odd_profiles': odd_profiles,
        'even_profiles': even_profiles,
        'min_allowed': min_allowed,
        'max_allowed': max_allowed
    }
    
    return results

def plot_SMI_results(results, bin_centers, reliable_cells=None, max_examples=6, exclude_boundary_cm=5):
    """
    Plot the results of SMI calculation with example cells.
    """
    SMI_values = results['SMI']
    preferred_positions = results['preferred_positions']
    non_preferred_positions = results['non_preferred_positions']
    valid_cells = results['valid_cells']
    odd_profiles = results['odd_profiles']
    even_profiles = results['even_profiles']
    min_allowed = results['min_allowed']
    max_allowed = results['max_allowed']
    
    # Apply reliability filter if provided
    if reliable_cells is not None:
        valid_cells = np.logical_and(valid_cells, reliable_cells)
    
    valid_indices = np.where(valid_cells)[0]
    n_valid = len(valid_indices)
    
    if n_valid == 0:
        print("No valid cells for SMI analysis.")
        return []
    
    print(f"Found {n_valid} valid cells for SMI analysis.")
    
    # Summary statistics
    valid_SMI = SMI_values[valid_cells]
    print(f"Mean SMI: {np.mean(valid_SMI):.3f}")
    print(f"Median SMI: {np.median(valid_SMI):.3f}")
    print(f"SMI range: {np.min(valid_SMI):.3f} to {np.max(valid_SMI):.3f}")
    # print(non_preferred_response[:10])
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
    inverted_modulated = np.sum(valid_SMI < -0.2)
    
    print(f"Strongly modulated (SMI > 0.5): {strongly_modulated} cells ({strongly_modulated/n_valid*100:.1f}%)")
    print(f"Moderately modulated (0.2 < SMI ≤ 0.5): {moderately_modulated} cells ({moderately_modulated/n_valid*100:.1f}%)")
    print(f"Weakly modulated (0 < SMI ≤ 0.2): {weakly_modulated} cells ({weakly_modulated/n_valid*100:.1f}%)")
    print(f"Non-modulated (|SMI| ≤ 0.05): {non_modulated} cells ({non_modulated/n_valid*100:.1f}%)")
    print(f"Inverted modulation (SMI < -0.2): {inverted_modulated} cells ({inverted_modulated/n_valid*100:.1f}%)")
    
    # Plot example cells
    # Sort by absolute SMI value to find most modulated cells
    sorted_indices = valid_indices[np.argsort(np.abs(valid_SMI))[::-1]]
    
    # Plot top examples
    n_examples = min(max_examples, n_valid)
    fig_examples = []
    
    for i in range(n_examples):
        cell_idx = sorted_indices[i]
        smi = SMI_values[cell_idx]
        pref_pos = preferred_positions[cell_idx]
        non_pref_pos = non_preferred_positions[cell_idx]
        
        # Create figure
        fig = plt.figure(figsize=(12, 5))
        
        # Plot odd and even trial profiles
        plt.subplot(1, 2, 1)
        plt.plot(bin_centers, odd_profiles[cell_idx], 'b-', label='Odd Trials (Training)')
        plt.plot(bin_centers, even_profiles[cell_idx], 'r-', label='Even Trials (Testing)')
        
        # Mark preferred and non-preferred positions
        plt.axvline(pref_pos, color='green', linestyle='--', alpha=0.7, 
                   label=f'Preferred Position ({pref_pos:.1f}cm)')
        plt.axvline(non_pref_pos, color='purple', linestyle='--', alpha=0.7, 
                   label=f'Non-Preferred Position ({non_pref_pos:.1f}cm)')
        
        # add vertical line at exclude_boundary_cm 
        # plt.axvline(min_allowed, color='red', linestyle='--', alpha=0.7)
        # plt.axvline(max_allowed, color='red', linestyle='--', alpha=0.7)
        # highlight the excluded boundary region
        plt.axvspan(0, min_allowed, color='red', alpha=0.1)
        # highlight the excluded boundary region from max_allowed to the end
        plt.axvspan(max_allowed, bin_centers[-1], color='red', alpha=0.1)
        

        plt.title(f'Cell {cell_idx} - SMI: {smi:.3f}')
        plt.xlabel('Position (cm)')
        plt.ylabel('Activity')
        plt.legend()
        
        # Plot response comparison at preferred and non-preferred positions
        plt.subplot(1, 2, 2)
        positions = ['Preferred', 'Non-Preferred']
        values = [results['Rp'][cell_idx], results['Rn'][cell_idx]]
        
        bars = plt.bar(positions, values, color=['green', 'purple'], alpha=0.6)
        
        plt.title(f'Response Comparison (Even Trials)')
        plt.ylabel('Response')
        
        
        # Add text with SMI value
        plt.text(0.5, max(values) * 1.1, f'SMI = {smi:.3f}', 
                horizontalalignment='center', fontsize=12)
        
        plt.tight_layout()
        fig_examples.append(fig)
    
    # Return all figures

    return [fig_hist] + fig_examples

def create_heatmap_visualization(results, spatial_activity, bin_centers, reliable_cells=None):
    """
    Create a heatmap visualization of cell responses sorted by SMI.
    """
    SMI_values = results['SMI']
    valid_cells = results['valid_cells']
    
    # Apply reliability filter if provided
    if reliable_cells is not None:
        valid_cells = np.logical_and(valid_cells, reliable_cells)
    
    valid_indices = np.where(valid_cells)[0]
    n_valid = len(valid_indices)
    
    if n_valid == 0:
        print("No valid cells for heatmap visualization.")
        return None
    
    # Sort cells by SMI
    sorted_indices = valid_indices[np.argsort(SMI_values[valid_indices])[::-1]]
    
    # Create matrix of trial-averaged activity for all valid cells
    activity_matrix = np.zeros((n_valid, len(bin_centers)))
    
    for i, cell_idx in enumerate(sorted_indices):
        # Get trial-averaged activity
        activity_matrix[i] = np.mean(spatial_activity[cell_idx], axis=0)
        
        # Normalize to 0-1 range for better visualization
        min_val = np.min(activity_matrix[i])
        max_val = np.max(activity_matrix[i])
        if max_val > min_val:
            activity_matrix[i] = (activity_matrix[i] - min_val) / (max_val - min_val)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(activity_matrix, aspect='auto', cmap='viridis',
                  extent=[bin_centers[0], bin_centers[-1], n_valid, 0])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Activity')
    
    # Add segment boundary
    corridor_length = np.max(bin_centers)
    segment_boundary = corridor_length / 2
    ax.axvline(segment_boundary, color='white', linestyle='--', alpha=0.7)
    
    # Add labels
    ax.set_title('Cells Sorted by Spatial Modulation Index (SMI)', fontsize=14)
    ax.set_xlabel('Position (cm)', fontsize=12)
    ax.set_ylabel('Cell Number (sorted by SMI)', fontsize=12)
    
    plt.tight_layout()
    return fig

def analyze_spatial_modulation(spatial_activity, bin_centers, reliable_cells=None, segment_distance=55, exclude_boundary_cm=3):
    """
    Simplified main function to perform spatial modulation analysis.
    
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
    print("=== SPATIAL MODULATION ANALYSIS ===")
    
    # 1. Calculate SMI
    print("Calculating Spatial Modulation Index (SMI)...")
    smi_results = calculate_SMI(
        spatial_activity, 
        bin_centers, 
        segment_distance=segment_distance,
        exclude_boundary_cm=exclude_boundary_cm
    )
    
    # 2. Plot SMI results
    print("Plotting SMI results...")
    smi_figures = plot_SMI_results(
        smi_results, bin_centers, reliable_cells, max_examples=50, exclude_boundary_cm=exclude_boundary_cm
    )
    
    # Display figures
    for fig in smi_figures:
        plt.figure(fig.number)
        plt.show()
    
    # # 3. Create heatmap visualization
    # print("Creating heatmap visualization...")
    # heatmap_fig = create_heatmap_visualization(
    #     smi_results, 
    #     spatial_activity, 
    #     bin_centers, 
    #     reliable_cells
    # )
    
    # if heatmap_fig:
    #     plt.figure(heatmap_fig.number)
    #     plt.show()
    
    print("Analysis complete!")
    
    return {
        'smi_results': smi_results,
        'smi_figures': smi_figures,
        # 'heatmap_fig': heatmap_fig
    }

# Example usage:
# results = analyze_spatial_modulation(normalized_spatial_activity, bin_centers, reliable_cells)