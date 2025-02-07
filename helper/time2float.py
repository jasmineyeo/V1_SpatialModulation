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
    