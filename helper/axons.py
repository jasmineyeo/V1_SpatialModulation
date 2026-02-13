"""
Utility functions for working with axonal two-photon calcium data.

It includes functions for identifying independent axons based on correlation coefficients,
removing correlated axons, and filtering dF/F traces.

Functions
---------
get_independent_axons(matpath, cc_thresh=0.5, gcc_thresh=0.5, apply_dFF_filter=False)
    Identifies independent axons from a .mat file containing calcium imaging data.

Author: DMM, last modified May 2025
"""


import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy import io
import itertools
from collections import defaultdict

from helper import correlation 
# import fm2p
from .twop import TwoP
import imgtools


def get_single_independent_axons(
        dFF, cc_thresh=0.5, gcc_thresh=0.5, apply_dFF_filter=False,
        fps=7.5, frame_means=None):
    """ Identify independent axons and drop the axons with the lower integrated dFF.
    
    Parameters
    ----------
    matpath : str
        Path to the .mat file containing calcium imaging data written by Matlab
        two-photon-calcium-post-processing pipeline (see README).
    cc_thresh : float, optional
        Threshold for between-cell correlation coefficient. Default is 0.5.
    gcc_thresh : float, optional
        Threshold for global frame correlation coefficient. Default is 0.5.
    apply_dFF_filter : bool, optional
        If True, apply a filter to the dF/F traces before calculating correlation
        coefficients. Default is False.

    Returns
    -------
    dFF_out : np.ndarray
        Filtered dF/F traces of independent axons.
    denoised_dFF : np.ndarray
        Denoised dF/F traces of independent axons.
    sps : np.ndarray
        Spike times of independent axons.
    usecells : list
        List of indices of independent axons.
    """

    if apply_dFF_filter:
        # smooth dFF traces
        all_smoothed_units = []
        for c in range(np.size(dFF, 0)):
            y = imgtools.nanmedfilt(
                    imgtools.rolling_average_1d(dFF[c,:], 11),
            25).flatten()
            all_smoothed_units.append(y)
        all_smoothed_units = np.array(all_smoothed_units)

    # calc between-cell correlation coeffients
    perm_mat = np.array(list(itertools.combinations(range(np.size(dFF, 0)), 2)))
    cc_vec = np.zeros([np.size(perm_mat,0)])
    if apply_dFF_filter:
        for i in range(np.size(perm_mat,0)):
            cc_vec[i] = correlation.corr2_coeff(
                all_smoothed_units[perm_mat[i,0]][np.newaxis,:],
                all_smoothed_units[perm_mat[i,1]][np.newaxis,:]
            )
    elif not apply_dFF_filter:
        for i in range(np.size(perm_mat,0)):
            cc_vec[i] = correlation.corr2_coeff(
                dFF[perm_mat[i,0]][np.newaxis,:],
                dFF[perm_mat[i,1]][np.newaxis,:]
            )

    # find axon pairs with cc above threshold
    check_index = np.where(cc_vec > cc_thresh)[0]
    exclude_inds = []

    for c in check_index:

        axon1 = perm_mat[c,0]
        axon2 = perm_mat[c,1]

        # exclude the neuron with the lower integrated dFF
        if (np.sum(dFF[axon1,:]) < np.sum(dFF[axon2,:])):
            exclude_inds.append(axon1)
        elif (np.sum(dFF[axon1,:]) > np.sum(dFF[axon2,:])):
            exclude_inds.append(axon2)

    exclude_inds = list(set(exclude_inds))
    usecells = [c for c in list(np.arange(np.size(dFF,0))) if c not in exclude_inds]

    if frame_means is not None:
        gcc_vec = np.zeros([len(usecells)])
        for i,c in enumerate(usecells):
            gcc_vec[i] = correlation.corr2_coeff(
                dFF[c,:][np.newaxis,:],
                frame_means
            )

        # find axons with gcc above threshold
        axon_correlates_with_globalF = np.where(gcc_vec > gcc_thresh)[0]
        usecells_gcc = [c for c in usecells if c not in axon_correlates_with_globalF]

        # remove axons with high correlation with global frame fluorescence
        dFF_out = dFF.copy()[usecells_gcc, :]
    
    elif frame_means is None:
        dFF_out = dFF.copy()[usecells, :]

    # remove axons w/ high correlation w/ other axons
    denoised_dFF, sps = TwoP.calc_inf_spikes(dFF_out, fps=fps)

    return dFF_out, denoised_dFF, sps, usecells



def get_grouped_independent_axons(
        dFF, cc_thresh=0.5, gcc_thresh=0.5, apply_dFF_filter=False,
        fps=7.5, frame_means=None
    ):
    """ Identify independent axons by grouping correlated sets and averaging
    their dFF traces.

    Parameters
    ----------
    matpath : str
        Path to the .mat file containing calcium imaging data written by Matlab
        two-photon-calcium-post-processing pipeline.
    cc_thresh : float, optional
        Threshold for between-cell correlation coefficient. Default is 0.5.
    gcc_thresh : float, optional
        Threshold for global frame correlation coefficient. Default is 0.5.
    apply_dFF_filter : bool, optional
        If True, apply a filter to the dF/F traces before calculating correlation
        coefficients. Default is False.

    Returns
    -------
    dFF_out : np.ndarray
        Averaged dF/F traces of independent axon groups.
    denoised_dFF : np.ndarray
        Denoised averaged dF/F traces.
    sps : np.ndarray
        Spike times of averaged independent axon groups.
    usecells : list of lists
        Each sublist contains indices of axons that were grouped and averaged
        into one trace.
    """

    # optionally smooth dFF
    if apply_dFF_filter:
        all_smoothed_units = []
        for c in range(np.size(dFF, 0)):
            y = imgtools.nanmedfilt(
                imgtools.rolling_average_1d(dFF[c, :], 11),
                25
            ).flatten()
            all_smoothed_units.append(y)
        all_smoothed_units = np.array(all_smoothed_units)
    else:
        all_smoothed_units = dFF

    # compute pairwise correlations
    perm_mat = np.array(list(itertools.combinations(range(np.size(dFF, 0)), 2)))
    cc_vec = np.zeros([np.size(perm_mat, 0)])
    for i in range(np.size(perm_mat, 0)):
        cc_vec[i] = correlation.corr2_coeff(
            all_smoothed_units[perm_mat[i, 0]][np.newaxis, :],
            all_smoothed_units[perm_mat[i, 1]][np.newaxis, :]
        )

    # build graph of correlated axons
    adjacency = defaultdict(set)
    for idx, c in enumerate(perm_mat):
        if cc_vec[idx] > cc_thresh:
            adjacency[c[0]].add(c[1])
            adjacency[c[1]].add(c[0])

    # find connected components (groups of correlated axons)
    visited = set()
    groups = []
    for node in range(np.size(dFF, 0)):
        if node not in visited:
            stack = [node]
            group = set()
            while stack:
                n = stack.pop()
                if n not in visited:
                    visited.add(n)
                    group.add(n)
                    stack.extend(adjacency[n] - visited)
            groups.append(sorted(list(group)))

    # average traces within each group
    averaged_traces = []
    for group in groups:
        avg_trace = np.mean(dFF[group, :], axis=0)
        averaged_traces.append(avg_trace)
    averaged_traces = np.array(averaged_traces)

    # Check correlation with global frame fluorescence
    # framef_ind = int(np.argwhere(np.asarray(mat['data'][0].dtype.names) == 'frame_F')[0])
    # frameF = mat['data'].item()[framef_ind].copy()

    if frame_means is not None:

        frame_means = frame_means[np.newaxis,:]

        gcc_vec = np.zeros([len(averaged_traces)])
        for i, trace in enumerate(averaged_traces):
            gcc_vec[i] = correlation.corr2_coeff(
                trace[np.newaxis, :],
                frame_means
            )

        # Keep only those groups not correlated with global fluorescence
        keep_inds = [i for i in range(len(averaged_traces)) if gcc_vec[i] <= gcc_thresh]

        dFF_out = averaged_traces[keep_inds, :]
        kept_groups = [groups[i] for i in keep_inds]

    elif frame_means is None:

        dFF_out = averaged_traces
        kept_groups = groups

    # Denoise and infer spikes
    denoised_dFF, sps = TwoP.calc_inf_spikes(dFF_out, fps=fps)

    return dFF_out, denoised_dFF, sps, kept_groups


def get_independent_axons(
        cfg, s2p_dict=None, matpath=None, merge_duplicates=True,
        cc_thresh=0.5, gcc_thresh=0.5, apply_dFF_filter=False
    ):

    fps = cfg['twop_rate']

    # if s2p_dict is not None:
    #     twop_data = fm2p.TwoP(cfg)
    #     twop_data.add_data(
    #         s2p_dict['F'],
    #         s2p_dict['Fneu'],
    #         s2p_dict['spks'],
    #         s2p_dict['iscell'],
    #     )
    #     twop_dict_out = twop_data.calc_dFF()
    #     dFF = twop_dict_out['raw_dFF']

    #     frame_means = twop_data.calc_frame_mean_across_time(
    #         s2p_dict['ops_path'],
    #         s2p_dict['bin_path']
    #     )
        
    if matpath is not None:
        mat = io.loadmat(matpath)
        try:
            dff_ind = int(np.argwhere(np.asarray(mat['data'][0].dtype.names)=='DFF')[0][0])
        except IndexError as e:
            print(e)
            print('There are no cells in this recording. Check cell segmentation.')
            quit()
        dFF = mat['data'].item()[dff_ind].copy()

        framef_ind = int(np.argwhere(np.asarray(mat['data'][0].dtype.names)=='frame_F')[0][0])
        frame_means = mat['data'].item()[framef_ind].copy().T

    if not merge_duplicates:
        # For each pair of correlated axons, drop the one with the lower integrated fluorescence
        return get_single_independent_axons(dFF, cc_thresh, gcc_thresh, apply_dFF_filter, fps=fps, frame_means=frame_means)
    
    elif merge_duplicates:
        # Instead of dropping one of each pair, merge them into a single axonal group, get the mean
        # dFF, and then calculate denoised dFF and inferred spikes using the merged dFF trace.
        # Probably the better approach
        return get_grouped_independent_axons(dFF, cc_thresh, gcc_thresh, apply_dFF_filter, fps=fps, frame_means=frame_means)