import xml.etree.ElementTree as ET
import datetime
import numpy as np
import h5py
from helper import time

def read_xml(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get the number of frames in the image stack
    nF = len(list(tree.find('Sequence')))


    # Absolute and relative times
    # Absolute are offset from the first timestamp by ~10 seconds
    absoluteT = np.zeros([nF], dtype=float)
    relativeT = np.zeros([nF], dtype=float)
    
    # Name of the file for each frame
    fnames = np.zeros([nF], dtype=str)

    # Iterate through the xml file and get the absolute and relative times
    for child in list(root.find('Sequence')):
        if child.tag != 'Frame':
            continue
        i = int(child.attrib['index'])-1

        _f = child.find('File').attrib['filename']

        absoluteT[i] = float(child.attrib['absoluteTime'])
        relativeT[i] = float(child.attrib['relativeTime'])
        fnames[i] = _f

    # Get the month, day and year from the xml file. This will be a
    # list of strings
    mdy = [int(x) for x in root.get('date').split(' ')[0].split('/')]

    # Get the starttime of the recording
    t0_str = root.find('Sequence').attrib['time']
    # Format that string into a datetime object with the correct day, month
    # and year, instead of the 1/1/1900 that it is create with by default
    t0 = (
        datetime.datetime.strptime(t0_str[:-1], '%H:%M:%S.%f')
        - datetime.datetime(year=1900, month=1, day=1)
        + datetime.datetime(mdy[2], mdy[0], mdy[1])
    )
    
    abs_2P_timestamps = absoluteT
    rel_2P_timestamps = relativeT
    num_2P_frames = nF
    
    # all_PVState_items = {}
    for i, child in enumerate(list((root.find('PVStateShard')))):
        child_items = [item for item in child.items()]
        if len(child_items) > 0:
            k = child_items[0][1]
            if k == 'framePeriod':
                acq_Hz = 1 / float(child_items[1][1])
            elif k == 'opticalZoom':
                optical_zoom = float(child_items[1][1])
            elif k == 'laserPower':
                laser_pockels = float(child[0].items()[1][1])
            elif k == 'laserWavelength':
                laser_wavelengths = float(child[0].items()[1][1])
            elif k == 'linesPerFrame':
                lines_per_frame = float(child.items()[1][1])
            elif k == 'micronsPerPixel':
                umPerPix_X = child[0].items()[1][1]
                umPerPix_Y = child[1].items()[1][1]
                umPerPix_Z = child[2].items()[1][1]
            elif k == 'pmtGain':
                pmt1_gain = child[0].items()[1][1]
                pmt2_gain = child[1].items()[1][1]
            elif k == 'positionCurrent':
                stage_position_X = child[0][0].items()[1][1]
                stage_position_Y = child[1][0].items()[1][1]
                stage_position_Z = child[2][0].items()[1][1]

    acq_props = {
        't0': t0,
        'abs_time': abs_2P_timestamps,
        'rel_time': rel_2P_timestamps,
        'num_frames': num_2P_frames,
        'acq_Hz': acq_Hz,
        'optical_zoom': optical_zoom,
        'laser_pockels': laser_pockels,
        'laser_wavelength': laser_wavelengths,
        'lines_per_frame': lines_per_frame,
        'um_per_pix_X': umPerPix_X,
        'um_per_pix_Y': umPerPix_Y,
        'um_per_pix_Z': umPerPix_Z,
        'pmt1_gain': pmt1_gain,
        'pmt2_gain': pmt2_gain,
        'stage_position_X': stage_position_X,
        'stage_position_Y': stage_position_Y,
        'stage_position_Z': stage_position_Z
    }

    return acq_props

def write_h5(filename, dic):
    """ Write a dictionary to an .h5 file.

    The dictionary can only contain values that are of the
    following types: dict, list, numpy.ndarray, or basic scalar
    types (int, float, str, bytes). The hierarchy of the dictionary
    is preserved in the .h5 file that is written. The keys of
    the dictionary can only be type str (not int).

    Modified from https://codereview.stackexchange.com/a/121308

    Parameters
    ----------
    filename : str
        Path to the .h5 file.
    dic : dict
        Dictionary to be saved.
    """

    with h5py.File(filename, 'w') as h5file:

        recursively_save_dict_contents_to_group(h5file, '/', dic)

def read_h5(filename, aslist=False):
    """ Read an .h5 file in as a dictionary.

    Modified from https://codereview.stackexchange.com/a/121308

    Parameters
    ----------
    filename : str
        Path to the .h5 file.
    aslist : bool
        If True, the dictionary will be read in as a list (on the first
        layer). Keys must have been convertable to integers when the file
        was written.
    """
    
    with h5py.File(filename, 'r') as h5file:

        out = recursively_load_dict_contents_from_group(h5file, '/')

        if aslist:

            outl = [None for l in range(len(out.keys()))]

            for key, item in out.items():
                outl[int(key)] = item
            out = outl


        return out
    
def recursively_save_dict_contents_to_group(h5file, path, dic):

    if isinstance(dic,dict):
        iterator = dic.items()

    elif isinstance(dic,list):
        iterator = enumerate(dic)

    else:
        ValueError('Cannot save %s type' % type(dic))

    for key, item in iterator:

        if isinstance(dic,list):
            key = str(key)
            
        if isinstance(item, (np.ndarray, np.int16, np.int64, np.float64,
                             int, float, str, bytes, np.float32, np.int32)):
            
            try:
                h5file[path + key] = item
            
            except TypeError:
                if isinstance(item, np.ndarray) and (item.dtype == object):
                    recursively_save_dict_contents_to_group(h5file, path + key + '/', item.item())

        elif isinstance(item, dict) or isinstance(item, list):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)

        elif isinstance(item, datetime.datetime):
             h5file[path + key] = time.time2str(item)

        else:
            raise ValueError('Cannot save %s type'%type(item))

def recursively_load_dict_contents_from_group(h5file, path):
    
    ans = {}

    for key, item in h5file[path].items():

        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]

        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file,
                                                                 path + key + '/')

    return ans
