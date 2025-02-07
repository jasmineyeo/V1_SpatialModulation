# main.py
import numpy as np
import os
from matplotlib import pyplot as plt
from helper import subroutine_find_corr, subroutine_test_r, suite2p2data_JSYEdit

# Input variables
animal_ID = int(input("Enter animal ID (omit initials): "))
first_repeat = int(input("Enter number of repeats in first session: "))
paired_movie = int(input("Enter number of the movie that opto was paired with: "))

# Get directoreis and data files
suite2pdata_directory, roiID, reliability_list_combined = data_directory(animal_ID, 0)

# Initialize variables
zplane = len(suite2pdata_directory)
session_num = len(suite2pdata_directory[0])
suite2p_filenames = np.full((zplane, session_num), "", dtype=object)
movOn_directory = np.empty((zplane, session_num), dtype=object)


# Process data
for z_num in range(zplane):
    for session in range(session_num):
        save_location = f"D:\\LongitudinalImaging_Data\\JSY0{animal_ID}\\Z{z_num + 1}"

        # Get directory info
        dir_info = os.listdir(suite2pdata_directory[z_num][session])
        time_filename = [f for f in dir_info if "timelog" in f]

        # Create movOn_directory path
        movOn_directory[z_num, session] = os.path.join(
            suite2pdata_directory[z_num][session], time_filename[0]
        )

        # Filename for saving
        filename = f"JSY0{animal_ID}_Z{z_num + 1}_session{session + 1}.mat"
        full_file_name = os.path.join(save_location, filename)

        # Check if the save location exists
        if os.path.exists(save_location):
            if os.path.exists(full_file_name):
                print(f"{full_file_name} file already exists")
            else:
                suite_2p_data = np.load(
                    os.path.join(suite2pdata_directory[z_num][session], "Fall.mat"),
                    allow_pickle=True,
                ).item()
                F = suite_2p_data["F"][suite_2p_data["iscell"][:, 0] == 1, :]
                Fneu = suite_2p_data["Fneu"][suite_2p_data["iscell"][:, 0] == 1, :]

                suite2p_processed_data = suite2p2data_JSYEdit(F, Fneu, 0, 0, 1)
                print(f"Running {filename}")

                # Save the processed data
                np.save(full_file_name, suite2p_processed_data)
        else:
            os.makedirs(save_location)
            suite_2p_data = np.load(
                os.path.join(suite2pdata_directory[z_num][session], "Fall.mat"),
                allow_pickle=True,
            ).item()
            F = suite_2p_data["F"][suite_2p_data["iscell"][:, 0] == 1, :]
            Fneu = suite_2p_data["Fneu"][suite_2p_data["iscell"][:, 0] == 1, :]

            suite2p_processed_data = suite2p2data_JSYEdit(F, Fneu, 0, 0, 1)
            print(f"Saving {filename}")

            # Save the processed data
            np.save(full_file_name, suite2p_processed_data)

            
# from scipy.io import loadmat, savemat
# from suite2p2data_JSYEdit import suite2p2data_JSYEdit

# suite_2p_data = loadmat(
#     "D:\\LongitudinalImaging_Data\\JSY026\\RawData\\NewPairing\\241222_JSY_JSY026_LI_D4\\TSeries-12222024-1430_Z1-001\\suite2p\\plane0\\Fall.mat"
# )
# F = suite_2p_data["F"][suite_2p_data["iscell"][:, 0] == 1, :]
# Fneu = suite_2p_data["Fneu"][suite_2p_data["iscell"][:, 0] == 1, :]

# suite2p_processed_data = suite2p2data_JSYEdit(F, Fneu, 0, 0, 1)
