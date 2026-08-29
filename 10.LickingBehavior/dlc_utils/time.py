# -*- coding: utf-8 -*-
"""
Time and timestamp helper functions.

Functions
---------
read_timestamp_series(s)
    Read timestamps as a pd.Series and format time.
interp_timestamps(camT, use_medstep=False)
    Interpolate timestamps for double the number of frames.
read_timestamp_file(timestamp_path, position_data_length=None, force_timestamp_shift=False)
    Read timestamps from a .csv file.
time2str(time_array)
    Convert datetime to string.
str2time(input_str)
    Convert string to datetime.
time2float(timearr, rel=None)
    Convert datetime to float.
interpT(x, xT, toT, fill_consecutive=False)
    Interpolate timestamps.
find_closest_timestamp(arr, t)
    Find the index of the closest timestamp to a given time.

Author: DMM, 2024
"""


import datetime
import scipy.interpolate
import pandas as pd
import numpy as np


def read_timestamp_series(s):
    """ Read timestamps as a pd.Series and format time.

    Parameters
    ----------
    s : pd.Series
        Timestamps as a Series. Expected to be formated as
        hours:minutes:seconds.microsecond

    Returns
    -------
    output_time : np.array
        Returned as the number of seconds that have passed since the
        previous midnight, with microescond precision, e.g. 700.000000

    """

    # Expected string format for timestamps.
    fmt = '%H:%M:%S.%f'

    output_time = []

    if s.dtype != np.float64:

        for current_time in s:

            str_time = str(current_time).strip()

            try:
                t = datetime.datetime.strptime(str_time, fmt)

            except ValueError as v:
                # If the string had unexpected characters (too much precision) for
                # one timepoint, drop the extra characters.

                ulr = len(v.args[0].partition('unconverted data remains: ')[2])
                
                if ulr:
                    str_time = str_time[:-ulr]
            
            try:
                output_time.append(
                        (datetime.datetime.strptime(str_time, '%H:%M:%S.%f')
                            - datetime.datetime.strptime('00:00:00.000000', '%H:%M:%S.%f')
                            ).total_seconds())

            except ValueError:
                output_time.append(np.nan)

        output_time = np.array(output_time)

    else:
        output_time = s.values

    return output_time


def interp_timestamps(camT, use_medstep=False):
    """ Interpolate timestamps for double the number of
    frames. Compensates for video deinterlacing.
    
    Parameters
    ----------
    camT : np.array
        Camera timestamps aquired at 30Hz
    use_medstep : bool
        When True, the median diff(camT) will be used as the timestep
        in interpolation. If False, the timestep between each frame
        will be used instead.

    Returns
    -------
    camT_out : np.array
        Timestamps of camera interpolated so that there are twice the
        number of timestamps in the array. Each timestamp in camT will
        be replaced by two, set equal distances from the original.

    """

    camT_out = np.zeros(np.size(camT, 0)*2)
    medstep = np.nanmedian(np.diff(camT, axis=0))

    if use_medstep:
        
        # Shift each deinterlaced frame by 0.5 frame periods
        # forward/backwards assuming a constant framerate

        camT_out[::2] = camT - 0.25 * medstep
        camT_out[1::2] = camT + 0.25 * medstep
    
    elif not use_medstep:

        # Shift each deinterlaced frame by the actual time between
        # frames. If a camera frame was dropped, this approach will
        # be more accurate than `medstep` above.
        
        steps = np.diff(camT, axis=0, append=camT[-1]+medstep)
        camT_out[::2] = camT
        camT_out[1::2] = camT + 0.5 * steps

    return camT_out


def read_timestamp_file(timestamp_path, position_data_length=None,
                        force_timestamp_shift=False):
    """ Read timestamps from a .csv file.

    Parameters
    ----------
    position_data_length : None or int
        Number of timesteps in data from deeplabcut. This is used to
        determine whether or not the number of timestamps is too short
        for the number of video frames.
        Eyecam and Worldcam will have half the number of timestamps as
        the number of frames, since they are aquired as an interlaced
        video and deinterlaced in analysis. To fix this, timestamps need
        to be interpolated.
    force_timestamp_shift : bool
        When True, the timestamps will be interpolated regardless of
        whether or not the number of timestamps is too short for the
        number of frames. Default is False.

    Returns
    -------
    camT : np.array
        Timestamps of camera interpolated so that there are twice the
        number of timestamps in the array than there were in the provided
        csv file.

    """

    # Read data and set up format
    s = pd.read_csv(timestamp_path, encoding='utf-8',
                    engine='c', header=None).squeeze()
    
    # If the csv file has a header name for the column, (which is
    # is the int 0 for some early recordings), remove it.
    if s[0] == 0:
        s = s[1:]
    
    # Read the timestamps as a series and format them
    camT = read_timestamp_series(s)
    
    # Auto check if vids were deinterlaced
    if position_data_length is not None:

        if position_data_length > len(camT):

            # If the number of timestamps is too short for the number
            # of frames, interpolate the timestamps.

            camT = interp_timestamps(camT, use_medstep=False)
    
    # Force the times to be shifted if the user is sure it should be done
    if force_timestamp_shift is True:

        camT = interp_timestamps(camT, use_medstep=False)
    
    return camT


def time2str(time_array):
    """ Convert datetime to string.

    The datetime values cannot be written into a hdf5
    file, so we convert them to strings before writing.

    Parameters
    ----------
    time_array : np.array, datetime.datetime
        If np.array with the shape (n,) where n is the
        number of samples in the recording. If datetime,
        the value will be converted to a single string.

    Returns
    -------
    out : str, list
        If time_array was a datetime, the returned value
        is a single string. Otherwise, it will be a list
        of strings with the same length as the input array.
        Str timestamps are use the format '%Y-%m-%d-%H-%M-%S-%f'.

    """

    fmt = '%Y-%m-%d-%H-%M-%S-%f'

    if type(time_array) == datetime.datetime:
        return time_array.strftime(fmt)


    out = []

    for t in time_array:
        tstr = t.strftime(fmt)
        out.append(tstr)

    return out


def str2time(input_str):
    """ Convert string to datetime.

    Need to convert the strings back to datetime objects
    after they are read back in from the hdf5 file.

    Parameters
    ----------
    input_str : str, byte, list, dict
        If str or byte, the value will be converted to a single
        datetime object. If list or dict, the values will be
        converted to an array of datetime objects. Datetime
        objects are returned with the format '%Y-%m-%d-%H-%M-%S-%f'.

    Returns
    -------
    out : datetime.datetime, np.array
        If input_str was a str or byte, the returned value is a single
        datetime object. Otherwise, it will be a np.array of datetime
        objects with the same length as the list or dict given for
        str_list.
    
    """

    fmt = '%Y-%m-%d-%H-%M-%S-%f'
    out = np.zeros(len(input_str), dtype=datetime.datetime)

    if type(input_str)==str:
        out = datetime.datetime.strptime(input_str, fmt)

    elif type(input_str)=='bytes':
        out = datetime.datetime.strptime(input_str.decode('utf-8'), fmt)

    elif type(input_str)==list:

        for i,t in enumerate(input_str):
            out[i] = datetime.datetime.strptime(t, fmt)

    elif type(input_str)==dict:

        for k,v in input_str.items():

            out[int(k)] = datetime.datetime.strptime(v.decode('utf-8'), fmt)

    return out


def time2float(timearr, rel=None):
    """ Convert datetime to float.

    Parameters
    ----------
    timearr : np.array
        Array of datetime objects.
    rel : datetime.datetime, optional
        If not None, the returned array will be relative
        to this time. The default is None, in which case the
        returned float values will be relative to the first
        time in timearr (i.e. start at 0 sec).
    
    Returns
    -------
    out : np.array
        Array of float values representing the time in seconds.
    
    """
    if rel is None:
        return [t.total_seconds() for t in (timearr - timearr[0])]
    elif rel is not None:
        if type(rel)==list or type(rel)==np.ndarray:
            rel = rel[0]
            rel = datetime.datetime(year=rel.year, month=rel.month, day=rel.day)
        return [t.total_seconds() for t in timearr - rel]


def interpT(x, xT, toT, fill_consecutive=False):
    """ Interpolate timestamps.
    
    Parameters
    ----------
    x : np.array
        Array of values to interpolate.
    xT : np.array
        Array of datetime objects corresponding to x.
    toT : np.array
        Array of datetime objects to interpolate to.

    Returns
    -------
    out : np.array
        Array of interpolated values.
    """

    # Convert timestamps to float values.
    if type(xT[0]) == datetime.datetime:
        xT = time2float(xT)
    if type(xT[0]) == datetime.datetime:
        toT = time2float(toT)

    # If the array is 1D and fill_consecutive is true, interpolate across NaNs and
    # fill NaNs forward and backward.
    if fill_consecutive and (len(np.shape(x))==1):
        x = pd.DataFrame(x.copy()).interpolate(limit=1, limit_direction='both').to_numpy().T[0]

    out = scipy.interpolate.interp1d(xT, x,
                   bounds_error=False)(toT)
    
    return out


def find_closest_timestamp(arr, t):
    """ Find the index of the closest timestamp to a given time.
    
    Parameters
    ----------
    arr : np.array
        Array of timestamps.
    t : float
        Time to find the closest timestamp to.

    Returns
    -------
    ind : int
        Index of the closest timestamp.
    approx_t : float
        Approximate timestamp value.
    """

    ind = np.nanargmin(np.abs(arr - t))
    approx_t = arr[ind]

    return ind, approx_t


def fmt_now(c=False):
    """Format today's date and time.

    Returns
    -------
    str_date : str
        Current date
        e.g. Aug. 30 2022 -> 220830
    str_time : str
        Current hours and minutes
        e.g. 10:15:00 am -> 10h-15m-00s
        Will be 24-hour time

    """
    str_date = datetime.datetime.today().strftime('%y%m%d')

    h = datetime.datetime.today().strftime('%H')
    m = datetime.datetime.today().strftime('%M')
    s = datetime.datetime.today().strftime('%S')
    str_time = '{}h-{}m-{}s'.format(h,m,s)

    if c==True:
        out = '{}_{}'.format(str_date, str_time)
        return out

    return str_date, str_time