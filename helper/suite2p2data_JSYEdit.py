# suite2p2data_JSYEdit.py

import numpy as np
from scipy.stats import gaussian_kde
from scipy.io import loadmat, savemat
# from subroutine import subroutine_test_r
from helper import subroutine_test_r

def suite2p2data_JSYEdit(F, Fneu, save_flag=1, deconcat_flag=0, num_envs=3):
    """
    This function converts the suite2p output to Goard Lab pipeline variables and performs
    the local neuropil subtraction.
    
    Parameters:
    - F: The fluorescence data (numpy array).
    - Fneu: The neuropil fluorescence data (numpy array).
    - save_flag: Whether to save the processed data (default is 1).
    - deconcat_flag: Flag for deconvolution, not used in this conversion (default is 0).
    - num_envs: Number of environments, only used for deconvolution (default is 3).
    
    Returns:
    - data: A dictionary with processed data.
    """

    # Check if the arguments are provided, otherwise load a default file.
    if F is None or Fneu is None:
        fall = loadmat('C:/Users/jasmineyeo/Desktop/B-1/240424_JSY_JSY020_LongitudinalImaging_B-1/TSeries-04242024-1407-005/suite2p/plane0/Fall.mat')
        F = fall['F']
        Fneu = fall['Fneu']

    # Prepare the data dictionary
    data = {}
    data['raw_F'] = F
    data['neuropil_F'] = Fneu

    # Local neuropil subtraction (assuming subroutine_test_r is a helper function that is also to be converted)
    print('Converting suite2p to data...')
    test_vec = np.arange(0, 1.01, 0.01)  # Equivalent to MATLAB's 0:0.01:1
    r_neuropil = subroutine_test_r(test_vec, data, 0)
    data['r_neuropil'] = r_neuropil

    DFF = np.zeros_like(data['raw_F'])

    # Loop over each cell (rows of the F matrix)
    for ii in range(data['raw_F'].shape[0]):
        raw_F = data['raw_F'][ii, :]

        # Find F0 using kernel density estimation (ksdensity in MATLAB)
        kde = gaussian_kde(raw_F)
        xi = np.linspace(np.min(raw_F), np.max(raw_F), 1000)  # KDE over a fine grid
        kde_values = kde(xi)
        max_idx = np.argmax(kde_values)
        F0 = xi[max_idx]

        # Raw DF/F calculation
        data['DFF_raw'] = np.zeros_like(data['raw_F'])
        data['DFF_raw'][ii, :] = (raw_F - F0) / F0 * 100

        # Subtract neuropil response
        neuropil_F = data['neuropil_F'][ii, :]
        norm_F = raw_F - r_neuropil[ii] * neuropil_F + r_neuropil[ii] * np.mean(neuropil_F)

        # Find F0 using mode of distribution estimate for normalized fluorescence
        kde_norm = gaussian_kde(norm_F)
        kde_norm_values = kde_norm(xi)
        max_idx_norm = np.argmax(kde_norm_values)
        F0_norm = xi[max_idx_norm]

        # DF/F calculation
        DFF[ii, :] = (norm_F - F0_norm) / F0_norm * 100

    data['DFF'] = DFF

    # Optionally save the processed data
    if save_flag:
        filename = 'Processed_Fall.mat'
        savemat(filename, {'data': data})
        
        # # Assuming spikeInference is another function to be translated
        # data = spikeInference(filename, 1)

        # # Save updated data
        # savemat(filename, {'data': data})

    # If deconcat_flag is set, additional functionality for deconvolution can be added here
    # if deconcat_flag:
    #     DeConcatenateEnvironments_v2(F, data, num_envs, frames)

    return data


# def spikeInference(filename, flag):
#     # This is a placeholder for spikeInference. You'll need to define the spike inference process here.
#     print(f"Running spike inference for {filename}")
#     return loadmat(filename)['data']  # Just a mock return, update as needed

