# -*- coding: utf-8 -*-
"""
landmarkGLM/config.py

DMM, Aug 2026
"""

import numpy as np


class Config:

    fps = 10.046
    fps_nominal = 10.0

    corridor_cm = 135.0

    # weird 2P vs VR clock offset. determined value empirically
    t_offset_s = -1.5

    zones = [(12.0, 37.0), (40.0, 65.0), (68.0, 93.0), (96.5, 121.5)]

    reward_gap_cm = 3.0

    neuropil_coef = 0.7
    baseline_pct = 8
    min_F0 = 25.0
    speed_thresh = 1.0
    iti_gap_s = 0.5
    min_stationary_s = 2.0

    glm_target = "spks"   # 'spks' only -- 'dff' is negative half the time
    zone_target = "spks"  # 'spks' only, same reason
    smooth_sigma_frames = 0 # 0 = no smoothing... smoothing inflates R2

    glm_rate_norm = True
    glm_mean_rate = 1.0

    glm_max_iter = 40
    glm_tol = 1e-7

    glm_backend = "auto"

    use_pure_behavior = False

    # lag basis
    lag_max_s = 2.0
    n_lag_basis = 7
    lag_log_offset = 0.05 # 'c' in log(tau + c); smaller = more front-loaded

    # cross val
    n_outer_folds = 20
    n_inner_folds = 3
    lambda_grid = np.logspace(0, 6, 9) # for the pure behavior block

    r2_threshold = 0.005
    free_lambda_sig_r2 = 0.02

    mu_grid = np.arange(2.5, 135.0, 5.0)
    delta_max_cm  = 8.0 # was 12.0
    delta_step_cm = 2.0
    delta_grid = np.arange(-delta_max_cm, delta_max_cm + 1e-9, delta_step_cm)
    sigma_grid = np.array([4.0, 7.0, 11.0, 16.0]) # bump widths, cm
    n_free_basis = 20 # unconstrained position basis = the ceiling

    kernel_lambdas = np.array([1.0, 10.0, 100.0])
    free_lambdas   = np.logspace(0, 5, 6)

    # penalty for position basis in pure-behavior model
    purebehavior_pos_scale = 4.0

    speed_span_cms  = (0.0, 28.0)
    accel_span_cms2 = (-50.0, 50.0)
    lapt_span_s     = (0.0, 35.0)
    rewt_span_s     = (0.0, 35.0)

    # Bases used only by the pure behavior block
    n_speed_basis = 8
    n_accel_basis = 6
    n_lapt_basis = 8
    n_rewt_basis = 8

    reward_gauss_in_PB = True

    reward_n_basis = 3 # bumps tiling the approach; 1 = single bump
    reward_span_cm = (122.0, 134.0) # centers are spread across this range
    reward_sigma_cm = 4.5 # cm, shared width
    reward_cut_cm = None # hard zero past here... None means corridor_cm

    onset_gauss_in_PB = True
    onset_n_basis = 2
    onset_span_cm = (5.0, 9.0)
    onset_sigma_cm = 2.0

    reward_impulse_in_PB = False
    n_reward_lag_basis = 8
    reward_lag_max_s = 8.0
    
    reward_lag_offset = 5.0
    
    linear_speed_accel = True   # one z-scored column each, not a value basis
    accel_in_PB = False
    onset_impulse_in_PB = False
    lap_time_in_PB = False
    reward_time_in_PB = False
    onset_gauss_in_PB = False

    reward_gauss_in_PB = True
    reward_impulse_in_PB = False

    purebehavior_pos_scale = 1.0

    va_slope_steps = None # None means len(delta_grid)
    va_slope_range = (0.20, 1.2)

    adapt_in_candidates = True
    adapt_ramp_s        = None # None means median lap duration
    adapt_per_lap_ramp  = False # True makes it distance-like
    adapt_match_scale   = True

    adapt_ramp_origin = "span"

    fit_span_cm = (12.0, 135.0)

    restrict_mu_to_span = True

    tx_vis_thr  = 0.20
    tx_conc_thr = 0.35

    shape_margin_min = 0.15
    shape_flat_z_min = 1.0

    pref_log_ratio_min = 0.25

    require_trace_agreement = True
