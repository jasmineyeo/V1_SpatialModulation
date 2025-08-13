import numpy as np
import datetime


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
