# -*- coding: utf-8 -*-
"""
landmarkGauss/gauss_config.py

DMM, Sept 2026
"""

import numpy as np

from config import Config


class GaussConfig(Config):

    gauss_bin_cm = 2.0
    gauss_fit_span_cm = None

    gauss_delta_max_cm = 12.0
    gauss_delta_step_cm = 0.5

    gauss_sigma_min_cm = 1.5
    gauss_sigma_max_cm = 12.0
    gauss_sigma_step_cm = 0.5

    # either 'fit' or 'percent'
    gauss_baseline_mode = "fit"
    gauss_baseline_pct = 10.0

    # polish the best grid point with bounded least squares (full-data fit only)
    gauss_refine = True
    
    gauss_n_splits = 20
    gauss_seed = 0

    # reliability thresholds
    gauss_rel_n_shuffles = 5
    gauss_rel_null_pct = 99.0
    gauss_rel_min_r = 0.50
    gauss_min_fit_r2 = 0.50

    # anything above this value is spaial, anything below or equal is visual.
    # this is just how well it does on the held-out data.
    gauss_visual_max_dr2 = 0.02

    def gauss_centers(self):
        return np.array([0.5 * (lo + hi) for lo, hi in self.zones], float)

    def gauss_span(self):
        if self.gauss_fit_span_cm is not None:
            return tuple(float(v) for v in self.gauss_fit_span_cm)
        gap = float(np.median([self.zones[j + 1][0] - self.zones[j][1]
                               for j in range(len(self.zones) - 1)]))
        return float(self.zones[0][0]) - gap, float(self.zones[-1][1]) + gap
