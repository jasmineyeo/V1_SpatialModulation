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

def load_data(twop_path, behav_path, xml_path=None):
        
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
    print(framerate)

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(im)
    ax1.set_title("raw_image")

    ax2.imshow(im_rotated)
    ax2.set_title("rotated_image")
    plt.show()

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

    # for any VR_data['location'] that is less than -40, set it to -40
    # VR_data['location'][VR_data['location'] < -40] = -40

    # Find the index of the first 's' in VR_data['event']
    start_index = np.where(VR_data['event'] == 's')[0][0]

    # Erase all elements before the start_index in all VR_data
    for key in VR_data.keys():
        VR_data[key] = VR_data[key][start_index:]

    # # for every element of VR_data, print first value
    # for key in VR_data:
    #     print(VR_data[key][0])

    print("first time value of VR is", VR_data['absoluteT'][0])
    print("first time value of 2p is", twoP_data['AbsoluteT'][0] )
    
