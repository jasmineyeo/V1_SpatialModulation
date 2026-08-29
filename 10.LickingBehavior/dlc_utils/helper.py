# -*- coding: utf-8 -*-
"""
Miscillaneous helper functions.

Functions
---------
split_xyl(xyl)
    Split the xyl dataframe into x, y, and likelihood dataframes.
apply_liklihood_thresh(x, l, threshold=0.99)
    Apply a likelihood threshold to a dataframe.
str_to_bool(value)
    Parse strings to read argparse flag entries in as bool.

Author: DMM, 2024
"""

import sys
import os
import pandas as pd
import numpy as np

import fm2p

def split_xyl(xyl):
    """ Split the xyl dataframe into x, y, and likelihood dataframes.
    
    Parameters
    ----------
    xyl : pd.DataFrame
        Dataframe containing x, y, and likelihood data.
    
    Returns
    -------
    x_vals : pd.DataFrame
        Dataframe containing x values.
    y_vals : pd.DataFrame
        Dataframe containing y values.
    l_vals : pd.DataFrame
        Dataframe containing likelihood values.
    """

    names = list(xyl.columns.values)

    x_locs = []
    y_locs = []
    l_locs = []

    # seperate the lists of point names into x, y, and likelihood
    for loc_num in range(0, len(names)):
        loc = names[loc_num]
        if '_x' in loc:
            x_locs.append(loc)
        elif '_y' in loc:
            y_locs.append(loc)
        elif 'likeli' in loc:
            l_locs.append(loc)

    x_vals = xyl[x_locs]
    y_vals = xyl[y_locs]
    l_vals = xyl[l_locs]

    return x_vals, y_vals, l_vals


def apply_liklihood_thresh(x, l, threshold=0.99):
    """ Apply a likelihood threshold to a dataframe.

    Parameters
    ----------
    x : pd.DataFrame
        Dataframe containing x or y values.
    l : pd.DataFrame
        Dataframe containing likelihood values.
    threshold : float, optional
        Likelihood threshold to apply. The default is 0.99.
    
    Returns
    -------
    x_vals : pd.DataFrame
        Dataframe containing x or y values with likelihood threshold applied. Values
        below the reshold are set to NaN.
    """

    thresh_arr = (l>threshold).astype(float).values
    x_vals1 = x.copy().values

    x_vals2 = pd.DataFrame((x_vals1 * thresh_arr), columns=x.columns)
    x_vals2[x_vals2==0.] = np.nan

    x_vals = x_vals2.copy()

    return x_vals


def str_to_bool(value):
    """ Parse strings to read argparse flag entries in as bool.
    
    Parameters
    ----------
    value : str
        Input value.

    Returns
    -------
    bool
        Input value as a boolean.
    """

    if isinstance(value, bool):
        return value
    
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    
    raise ValueError(f'{value} is not a valid boolean value')


def make_default_cfg():
    internals_config_path = os.path.join(fm2p.up_dir(__file__, 1), 'internals.yaml')
    cfg = fm2p.read_yaml(internals_config_path)

    return cfg

def to_dict_of_arrays(df):
    seriesdict = {}
    for key in df.keys():
        seriesdict[key] = df[key].to_numpy()
    return seriesdict

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def fix_dict_dtype(d, totype):
    
    for k,v in d.items():
        if type(v) == dict:
            d[k] = fix_dict_dtype(d[k], totype)
            continue
        if type(v) == list:
            d[k] = [x.astype(totype) for x in v]
            continue
        if type(v) == np.ndarray:
            d[k] = v.astype(totype).tolist()
            continue
        d[k] = float(v)

    return d


def nan_filt(items):
    # 'items' must be a list of arrays or list-like objects

    if any([type(arr)!=np.ndarray for arr in items]):
        items = [np.array(arr) for arr in items]

    shapes = [arr.shape for arr in items]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError('All input arrays must have the same shape.')
    
    assert items[0].ndim == 2

    mask = ~np.isnan(np.vstack(items)).any(axis=0)
    items_out = [arr[:, mask] for arr in items]

    return items_out
