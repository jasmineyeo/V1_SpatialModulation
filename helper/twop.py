"""
helper/twop.py
Two-photon imaging data analysis class

DMM, 2024
"""

import os
import json
import numpy as np
import scipy.stats
import oasis

class TwoP():

    def __init__(self, recording_path, recording_name, cfg=None, props=None, rnum=np.nan):
        
        self.recording_path = recording_path
        self.recording_name = recording_name

        if cfg is None:
            self.twop_dt = 1./10.

        if (props is not None) and (~np.isnan(rnum)):
            with open(props, 'r') as f:
                session_props = json.load(f)
            self.rstr = 'R{:02}'.format(rnum)
            rdir = session_props[self.rstr]['rec_dir']

            self.suite2p_path = os.path.join(rdir, session_props[self.rstr]['suite2p'])

            self.suite2p_outputs = np.load(self.suite2p_path)
            self.F = self.suite2p_outputs['F']
            self.Fneu = self.suite2p_outputs['Fneu']
            iscell = self.suite2p_outputs['iscell']
            self.s2p_spks = self.suite2p_outputs['spks']

            usecells = iscell[:,0]==1

            self.F[usecells, :]
            self.Fneu[usecells, :]
            self.s2p_spks[usecells, :]

    def find_files(self):

        self.F = np.load(os.path.join(self.recording_path, r'suite2p/plane0/F.npy'), allow_pickle=True)
        self.Fneu = np.load(os.path.join(self.recording_path, r'suite2p/plane0/Fneu.npy'), allow_pickle=True)
        iscell = np.load(os.path.join(self.recording_path, r'suite2p/plane0/iscell.npy'), allow_pickle=True)
        spks = np.load(os.path.join(self.recording_path, r'suite2p/plane0/spks.npy'), allow_pickle=True)

        usecells = iscell[:,0]==1

        self.F = self.F[usecells, :]
        self.Fneu = self.Fneu[usecells, :]
        self.s2p_spks = spks[usecells, :]


    def calc_dFF(self, neu_correction=0.7):

        F = self.F
        Fneu = self.Fneu

        nCells, lenT = np.shape(F)

        norm_F = np.zeros([nCells, lenT])
        raw_dFF = np.zeros([nCells, lenT])
        norm_dFF = np.zeros([nCells, lenT])
        norm_F0 = np.zeros(nCells)
        raw_F0 = np.zeros(nCells)
        denoised_dFF = np.zeros([nCells, lenT])
        sps = np.zeros([nCells, lenT])

        for c in range(nCells):
            
            F_cell = F[c,:].copy()
            F_cell_neu = Fneu[c,:].copy()

            _f0_raw = scipy.stats.mode(F_cell, nan_policy='omit').mode

            # Raw DF/F
            _raw_dFF = (F_cell - _f0_raw) / _f0_raw * 100

            # Subtract neuropil
            _normF = F_cell - neu_correction * F_cell_neu + neu_correction * np.nanmean(F_cell_neu)

            _f0_norm = scipy.stats.mode(_normF, nan_policy='omit').mode

            # dF/F with neuropil correction
            norm_dFF[c,:] = (_normF - _f0_norm) / _f0_norm * 100

            # below lines (99-101) commented out because oasis package could not be installed (JSY - 02/03/2025)
            # deconvolved spiking activity and denoised fluorescence signal
            g = oasis.functions.estimate_time_constant(norm_dFF[c,:].copy(), 1)
            denoised_dFF[c,:], sps[c,:] = oasis.oasisAR1(norm_dFF[c,:].copy(), g)

            norm_F[c,:] = _normF
            raw_dFF[c,:] = _raw_dFF
            norm_F0[c] = _f0_norm
            raw_F0[c] = _f0_raw

        twop_dict = {
            'raw_F0': raw_F0,
            'norm_F0': norm_F0,
            'raw_F': F,
            'norm_F': norm_F,
            'raw_Fneu': Fneu,
            'raw_dFF': raw_dFF,
            'norm_dFF': norm_dFF,
            'denoised_dFF': denoised_dFF,
            'spikes_per_sec': sps,
            's2p_spks': self.s2p_spks
        }

        return twop_dict


    def save_fluor(self, twop_dict):

        savedir = os.path.join(self.recording_path, self.recording_name)
        _savepath = os.path.join(savedir, '{}_twophoton.h5'.format(self.recording_name))
        # fm2p.write_h5(_savepath, twop_dict)