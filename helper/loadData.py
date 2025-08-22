"""
helper/preprocessData.py
preprocess two-photon imaging data and treadmill behavior data for laminar analysis

"""

import os
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
from helper import TwoP, read_xml, time2float

class dataLoader:
    def __init__(self, twop_path, behav_path):
        self.twop_path = twop_path
        self.behav_path = behav_path
        
    def load_data(self):
        twop_path = self.twop_path
        behav_path = self.behav_path
        
        # define a twoP_filename which is the variable after the very last \ from the twop_path
        twoP_filename = os.path.basename(twop_path)
        behav_filename = os.path.basename(behav_path)

        # Extract animal ID and date from the VR_log_filename
        match = re.match(r"VRlog_(JSY\d+)_(\d{8})_\d{2}-\d{2}-\d{2}\.txt", behav_filename)
        if match:
            animal_id = match.group(1)
            date = match.group(2)
        else:
            print("Filename format does not match the expected pattern.")

        # Initialize dictionaries to store raw data
        twoP_data = {}
        VR_data = {}

        # Load twoP data
        raw_twop_data = TwoP(twop_path, twoP_filename)

        raw_twop_data.find_files()
        twop_dict = raw_twop_data.calc_dFF()

        twoP_data['sps'] = twop_dict['spikes_per_sec'].copy()
        twoP_data['s2p_spks'] = twop_dict['s2p_spks'].copy()
        twoP_data['dFF'] = twop_dict['norm_dFF'].copy()
        twoP_data['stat'] = twop_dict['stat'].copy()
        twoP_data['ops'] = twop_dict['ops'].copy()

        numFrames = np.size(twoP_data['sps'], 1)
        numCells = len(twoP_data['stat'])

        xml_path = os.path.join(twop_path, f"{twoP_filename}.xml")
        xml_dict = read_xml(xml_path)
        t0 = xml_dict["t0"]
        abs_time = xml_dict["abs_time"]
        rel_time = xml_dict["rel_time"]
        framerate = 1/rel_time[1]

        twopT = np.zeros(np.size(abs_time, 0) - 1, dtype=datetime.datetime)
        for rep, t in enumerate(abs_time[:-1]):
            twopT[rep] = t0 + datetime.timedelta(seconds=t)

        twopT_float = time2float(twopT)
        twoP_data['AbsoluteT'] = twopT

        im = np.zeros((twoP_data['ops']['Ly'], twoP_data['ops']['Lx']))  # Create an empty image
        for n in range(0, numCells):
            ypix = twoP_data['stat'][n]['ypix'][~twoP_data['stat'][n]['overlap']]
            xpix = twoP_data['stat'][n]['xpix'][~twoP_data['stat'][n]['overlap']]
            im[ypix, xpix] = xpix  # Assign xpix values to im for progressive color change along x-axis

        # for animal facing 2p computer, image should be rotated so it goes from layer 2/3 to layer 6 (top-bottom)
        # for animal facing VR computer, raw image does go from layer 2/3 to layer 6 (top-bottom)
        im_rotated = np.rot90(im, k=-1)

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # ax1.imshow(im)
        # ax1.set_title("raw_image")

        # ax2.imshow(im_rotated)
        # ax2.set_title("rotated_image")
        # plt.show()

        # Load VRlog
        rawVR_data = []
        with open(behav_path, "r") as file:
            lines = file.readlines()
            for line in lines[3:]:
                rawVR_data.append(line.strip().split("\t"))

        # Extract VR data
        VR_data['absoluteT'] = np.array([line[0] for line in rawVR_data])
        VR_data['elapsedT'] = np.array([float(line[1]) for line in rawVR_data])
        VR_data['event'] = np.array([line[2] for line in rawVR_data])
        VR_data['location'] = np.array([float(line[3]) for line in rawVR_data])

        # for any VR_data['location'] that is less than 0, set it to 0
        VR_data['location'][VR_data['location'] < 0] = 0

        # Find the index of the first 's' in VR_data['event']
        start_index = np.where(VR_data['event'] == 's')[0][0]

        # Erase all elements before the start_index in all VR_data
        for key in VR_data.keys():
            VR_data[key] = VR_data[key][start_index:]

        # # for every element of VR_data, print first value
        # for key in VR_data:
        #     print(VR_data[key][0])

        print("first time value of VR is", VR_data['absoluteT'][0])
        print("first time value of 2p is", twoP_data['AbsoluteT'][0])
        
        self.twoP_data = twoP_data
        self.VR_data = VR_data
        
        return animal_id, date, framerate


    def align_data(self):
        # Align the twoP and VR data based on their timestamps
        twoP_data = self.twoP_data
        VR_data = self.VR_data
        
        # Define absolute_t0 as the first element of VR_data['absoluteT'] -- with "s" for event type, which is the timestamp for 2p input trigger
        VR_absolute_t = np.array([datetime.datetime.strptime(t, '%H.%M.%S.%f') for t in VR_data['absoluteT'][0:]])

        # Calculate relative_t (time elapsed from absolute_t0)
        VR_relative_t = np.array([(t - VR_absolute_t[0]).total_seconds() for t in VR_absolute_t])

        # Add twoP_data['AbsoluteT'][0] to each timedelta object to get vrT
        VR_relative_t_timedelta = np.array([datetime.timedelta(seconds=t) for t in VR_relative_t])
        Aligned_Abs_vrT = twoP_data['AbsoluteT'][0] + VR_relative_t_timedelta

        # Find the closest value in Aligned_Abs_vrT that is greater than twoP_data['AbsoluteT'][-1]
        closest_value = Aligned_Abs_vrT[Aligned_Abs_vrT > twoP_data['AbsoluteT'][-1]][0]
        closest_index = np.where(Aligned_Abs_vrT == closest_value)[0][0]

        new_VR_data = {}
        new_VR_data['AbsoluteT'] = np.array(Aligned_Abs_vrT)[:closest_index]
        new_VR_data['RelativeT'] = VR_relative_t[:closest_index]
        new_VR_data['event'] = VR_data['event'][:closest_index]
        new_VR_data['location'] = VR_data['location'][:closest_index]

        # Calculate relative time points for VR_data and twoP_data
        twop_relativeT = twoP_data['AbsoluteT'] - twoP_data['AbsoluteT'][0]

        # Convert to seconds
        twop_relativeT = np.array([t.total_seconds() for t in twop_relativeT])
        twoP_data['RelativeT'] = twop_relativeT

        # Interpolate the location at twoP_data['RelativeT'] from new_VR_data['location'] at new_VR_data['RelativeT']
        interpolated_location = np.interp(twoP_data['RelativeT'], 
                                        new_VR_data['RelativeT'], 
                                        new_VR_data['location'])
        new_VR_data['interp_location'] = interpolated_location
        print(f"size of interpolated_location is {interpolated_location.shape}")
        print(f"size of new_VR_data['location'] is {new_VR_data['location'].shape}")

        # Plot the interpolated location
        # plt.figure(figsize=(20, 5))
        # plt.plot(twoP_data['RelativeT'], new_VR_data['interp_location']+100, label="Interpolated Location", alpha=0.5)
        # plt.plot(new_VR_data['RelativeT'], new_VR_data['location'], label="Original Location", alpha=0.5)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Location (AU)")
        # plt.legend()
        # plt.show()

        self.new_VR_data = new_VR_data

        return twoP_data, new_VR_data