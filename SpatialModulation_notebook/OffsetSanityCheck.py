"""
OffsetSanityCheck.py
A script checking timing between 2p and VR
Using 2photon data with Unity rendering an illuminating sphere object
Input: average fluourescent data in a .mat format

JSY, 09/02/25
"""

import os
import re
import scipy.io as sio
import numpy as np
import datetime
from helper import TwoP, read_xml, time2float, SpikeSmoothing


# load data

# define a twoP_filename which is the variable after the very last \ from the twop_path
twop_path = r"F:\2P\spmod\250829_JSY_VR_timingcheck\TSeries-08292025-1424.1.15gain-001"
behav_path = r"..."

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

# Load .mat file
twoP_data['fluorescence'] = sio.loadmat(os.path.join(twop_path, "gain1.15_averageF.mat"))
numFrames = np.size(twoP_data['fluorescence'], 1)
numCells = len(twoP_data['fluorescence'])

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


# align data

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


optimal_offset, _, _ = SpikeSmoothing.run_offset_optimization(twop_filepath, vr_filepath)
