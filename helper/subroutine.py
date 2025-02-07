# subroutine.py
import numpy as np
import matplotlib.pyplot as plt

def subroutine_find_corr(r_neuropil, data, print_flag=0):
    """
    Finds the correlation coefficient for a specific r_neuropil value.
    Called by subroutine_test_r.

    Parameters:
    - r_neuropil: The current r_neuropil value to test.
    - data: Dictionary containing 'raw_F' and 'neuropil_F'.
    - print_flag: If 1, print the average correlation.

    Returns:
    - final_corr: Array of correlation coefficients (squared) for each cell.
    """
    # Calculate corrected fluorescence
    test_F = data['raw_F'] - r_neuropil * data['neuropil_F']

    # Initialize vector for correlation coefficients
    corr = np.zeros(data['raw_F'].shape[0])

    # Calculate squared correlation coefficient for each cell
    for i in range(data['raw_F'].shape[0]):
        # Calculate correlation coefficient matrix
        corr_matrix = np.corrcoef(test_F[i, :], data['neuropil_F'][i, :])
        # Extract the correlation coefficient and square it
        corr[i] = corr_matrix[0, 1] ** 2

    # Return final correlation coefficients
    final_corr = corr

    # Print the average correlation if flag is set
    if print_flag == 1:
        print(f"Average correlation = {np.mean(corr)}")

    return final_corr

def subroutine_test_r(test_vec, data, plot_flag=0):
    """
    Tests different r_neuropil values to find the value that minimizes the
    correlation between the corrected response and the neuropil.

    Parameters:
    - test_vec: Array of r_neuropil values to test.
    - data: Dictionary containing 'raw_F' and 'neuropil_F' data.
    - plot_flag: If 1, plot the mean correlation against test_vec.

    Returns:
    - min_corr: Array of r_neuropil values that minimize correlation for each cell.
    """
    # Initialize matrix of correlation coefficients (cell x r_neuropil)
    corr_mat = np.zeros((data['raw_F'].shape[0], len(test_vec)))

    # Compute correlation matrix
    for i, r in enumerate(test_vec):
        corr_mat[:, i] = subroutine_find_corr(r, data, 0)  # Fill column with correlation values

    # Handle NaN values in correlation matrix
    corr_mat[np.isnan(corr_mat)] = 0

    # Find the index of the r_neuropil that minimizes the mean correlation
    mean_corr = np.mean(corr_mat, axis=0)
    idx = np.argmin(mean_corr)
    optimal_r = test_vec[idx]

    if plot_flag:
        plt.plot(test_vec, mean_corr)
        plt.xlabel('r_neuropil')
        plt.ylabel('Mean Correlation')
        plt.title('Mean Correlation vs. r_neuropil')
        plt.show()

    # Find the r_neuropil value that minimizes correlation for each cell
    min_idx = np.argmin(corr_mat, axis=1)
    min_corr = np.array([test_vec[idx] for idx in min_idx])

    print(f"Mean r_neuropil = {np.mean(min_corr)}")
    return min_corr