# Multi_Naturalmovie_code_cohensd_suite2p
import numpy as np
from scipy.io import loadmat
def multi_naturalmovie_code_cohensd_suite2p(filename1, filename2, frame_rate, cohen_thresh, CCprt):   
    # Load data
    # check if inputs are entered
    # if not filename1:
    #     # allow the user to enter the filename
        
    #     print('Please enter a Suite2p data file')
    #     # let the user enter the directory of filename1
        
    #     return 
    suite_2p_data = np.load(filename1, allow_pickle=True)
    mov_on = loadmat(filename2)
    pres_order = loadmat(r'D:\LongitudinalImaging_Data\JSY026\2_moviepres_order.mat')
    
    print('Loading stimuli presentation order...')
    
    # Parameters
    repeats = 10
    on_time = 20
    shuffles = 1000
    num_stim = mov_on['mov_on'].shape[0]
    
    # Calculate additional parameters
    on_frames = int(round(on_time * frame_rate))
        
    # Initialize matrices
    num_cells = np.size(suite_2p_data['DFF'], 0)
    sorted_pres_order = np.zeros_like(pres_order)
    sorted_mov_on = np.zeros_like(mov_on['mov_on'])
    resp_vec = np.zeros((repeats, on_frames, num_cells, num_stim))
    
    # Sort mov_on by sorted pres_order
    for i in range(repeats):
        sorted_indices = np.argsort(pres_order[:, i])
        sorted_pres_order[:, i] = pres_order[sorted_indices, i]
        sorted_mov_on[:, i] = mov_on['mov_on'][sorted_indices, i]
    
    # Create response vector and sort DFF data
    for stim in range(num_stim):
        for rep in range(repeats):
            curr_frame = int(round(sorted_mov_on[stim, rep] * frame_rate))
            resp_vec[rep, :, :, stim] = suite_2p_data['suite2pProcessedData']['DFF'][:, curr_frame:curr_frame + on_frames].T
    
    # Test reliability of each cell
    print('Testing reliability of each cell...')
    
    iterated_CC = np.zeros((shuffles, num_cells, num_stim))
    average_CC = np.zeros((num_stim, num_cells))
    reliable_cell_vec = np.zeros((num_stim, num_cells))
    
    for stim in range(num_stim):
        for cell in range(num_cells):
            bt_CC_data = np.zeros(shuffles)
            bt_CC_rand = np.zeros(shuffles)
            
            for shuffle in range(shuffles):
                activity_data = resp_vec[:, :, cell, stim]
                activity_rand = np.array([
                    np.roll(activity_data[rep], np.random.randint(on_frames))
                    for rep in range(repeats)
                ])
                
                # Randomly split laps into two halves
                trial_shuffle = np.random.permutation(repeats)
                trial_select1 = trial_shuffle[:repeats // 2]
                trial_select2 = trial_shuffle[repeats // 2:]
                
                # Calculate CC from two halves of data
                first_half_mean = np.mean(activity_data[trial_select1], axis=0)
                second_half_mean = np.mean(activity_data[trial_select2], axis=0)
                corr_matrix = np.corrcoef(first_half_mean, second_half_mean)
                bt_CC_data[shuffle] = corr_matrix[0, 1]
                iterated_CC[shuffle, cell, stim] = corr_matrix[0, 1]
                
                # Calculate CC from two halves of randomized data
                first_half_rand_mean = np.mean(activity_rand[trial_select1], axis=0)
                second_half_rand_mean = np.mean(activity_rand[trial_select2], axis=0)
                corr_rand_matrix = np.corrcoef(first_half_rand_mean, second_half_rand_mean)
                bt_CC_rand[shuffle] = corr_rand_matrix[0, 1]
            
            average_CC[stim, cell] = np.mean(iterated_CC[:, cell, stim])
            
            # Test actual CC distribution against shuffled distribution
            avg_bt_CC_data = np.mean(bt_CC_data)
            shuff_interval = np.percentile(bt_CC_rand, CCprt)
            
            x1 = bt_CC_data
            x2 = bt_CC_rand
            n1, n2 = len(x1), len(x2)
            mean_diff = np.mean(x1) - np.mean(x2)
            pooled_sd = np.sqrt(((n1 - 1) * np.var(x1) + (n2 - 1) * np.var(x2)) / (n1 + n2 - 2))
            d = mean_diff / pooled_sd  # Cohen's d
            
            if avg_bt_CC_data > shuff_interval and d > cohen_thresh:
                reliable_cell_vec[stim, cell] = 1
    
    print(f'Total number of Responsive Cells to Stim 1 = {np.sum(reliable_cell_vec[0, :]) / num_cells * 100:.2f}%')
    print(f'Number of Responsive Cells to Stim 1 = {np.sum(reliable_cell_vec[0, :])}')
    print(f'Mean CC for Responsive Cells - Stim 1 = {np.mean(average_CC[0, reliable_cell_vec[0, :] == 1])}')
    
    print(f'Total number of Responsive Cells to Stim 2 = {np.sum(reliable_cell_vec[1, :]) / num_cells * 100:.2f}%')
    print(f'Number of Responsive Cells to Stim 2 = {np.sum(reliable_cell_vec[1, :])}')
    print(f'Mean CC for Responsive Cells - Stim 2 = {np.mean(average_CC[1, reliable_cell_vec[1, :] == 1])}')
    
    return {
        'RespVec': resp_vec,
        'reliable_cell_vec': reliable_cell_vec,
        'average_CC': average_CC
    }