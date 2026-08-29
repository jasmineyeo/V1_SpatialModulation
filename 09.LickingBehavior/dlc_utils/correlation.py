# -*- coding: utf-8 -*-
"""
Correlation helper functions.

Functions
---------
nanxcorr(x, y, maxlag=25)
    Cross correlation ignoring NaNs.
corr2_coeff(A, B)
    Calculate the correlation coefficient between two 2D arrays.

Author: DMM, 2025
"""
import numpy as np
import pandas as pd


def nanxcorr(x, y, maxlag=25):
    """ Cross correlation ignoring NaNs.

    Parameters
    ----------
    x : array
        Array of values.
    y : array
        Array of values to shift. Must be same length as x.
    maxlag : int
        Number of lags to shift y prior to testing correlation.
    
    Returns
    -------
    cc_out : array
        Cross correlation.
    lags : range
        Lag vector.
    """

    lags = range(-maxlag, maxlag)
    cc = []

    for i in range(0,len(lags)):
        
        # shift data
        yshift = np.roll(y, lags[i])
        
        # get index where values are usable in both x and yshift
        use = ~pd.isnull(x + yshift)
        
        # some restructuring
        x_arr = np.asarray(x, dtype=object)
        yshift_arr = np.asarray(yshift, dtype=object)

        x_use = x_arr[use]
        yshift_use = yshift_arr[use]
        
        # normalize
        x_use = (x_use - np.mean(x_use)) / (np.std(x_use) * len(x_use))

        yshift_use = (yshift_use - np.mean(yshift_use)) / np.std(yshift_use)
        
        # get correlation
        cc.append(np.correlate(x_use, yshift_use))

    cc_out = np.hstack(np.stack(cc))
    
    return cc_out, lags


def corr2_coeff(A, B):
    """ Calculate the correlation coefficient between two 2D arrays.

    This is more efficient than scipy methods for calculating Pearson correlation,
    especially for large arrays.

    Parameters
    ----------  
    A : np.ndarray
        2D array of values.
    B : np.ndarray
        2D array of values to compare. Must be same shape as A.
    
    Returns
    -------
    corr_coeff : float
        Correlation coefficient between A and B.
    """


    # Row-wise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    corr_coeff = np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))
    return corr_coeff[0][0]

