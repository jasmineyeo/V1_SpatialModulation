# -*- coding: utf-8 -*-
"""
landmarkGLM/save.py

DMM, Aug 2026
"""

import os
import pickle

import numpy as np
import pandas as pd

CFG_KEYS = ["corridor_cm", "zones", "fps", "zone_target",
            "n_outer_folds", "r2_threshold", "n_lag_basis",
            "lag_max_s", "linear_speed_accel", "accel_in_PB",
            "onset_impulse_in_PB",
            "lap_time_in_PB", "reward_time_in_PB",
            "reward_gauss_in_PB", "reward_n_basis", "reward_span_cm",
            "reward_sigma_cm", "onset_gauss_in_PB", "onset_n_basis",
            "onset_span_cm", "onset_sigma_cm",
            "reward_impulse_in_PB", "n_reward_lag_basis",
            "reward_lag_max_s", "purebehavior_pos_scale",
            "free_lambda_sig_r2", "speed_span_cms",
            "accel_span_cms2", "lapt_span_s", "rewt_span_s",
            "adapt_in_candidates", "adapt_ramp_s", "adapt_per_lap_ramp",
            "adapt_match_scale"]


def save_outputs(km, beh, cell_ids, cfg, outdir, rel=None, mod=None):

    cols = {
        "suite2p_cell_id": cell_ids,
        "r2_one_peak": km["gauss"],
        "r2_four_peak": km["comb"],
        "r2_both": km["both"],
        "r2_free_ceiling": km["free"],
        "place_index": km["place_index"], # +1 spatial, -1 visual
        "peak_position_cm": km["mu"],
        "peak_sigma_cm": km["sigma"],
        "peak_stability_cm": km["mu_stability"],
        "comb_offset_cm": km["delta"],
        "spatial_index": km["spatial_index"],
        "best_model": km["best_model"],
        "best_r2": km["best_r2"],
        "reliable": km["reliable"],
        "fit_above_null": km["fit_above_null"],
        "well_fit": km["well_fit"],
    }
    if "adapt" in km:
        cols["r2_adapt"] = km["adapt"]
        cols["r2_adapt_gain"] = km["adapt_gain"]
        for k in ("adapt_index", "visual_r2"):
            if k in km:
                cols[k] = km[k]
        cols["adaptation_index"] = km["adaptation_index"]
        cols["adaptation_base"] = km["adaptation_base"]
        cols["adapt_rate"] = km["adapt_rate"]
        cols["adapt_offset_cm"] = km["adapt_delta"]
        cols["adapt_sigma_cm"] = km["adapt_sigma"]
    if rel is not None:

        cols["split_half_r"] = rel["r"]
    if mod is not None:

        cols["onset_modulation"] = mod["onset_mi"]
        cols["reward_modulation"] = mod["reward_mi"]
        cols["has_onset_response"] = mod["has_onset"]
        cols["has_reward_response"] = mod["has_reward"]

    tab = pd.DataFrame(cols)
    csv = os.path.join(outdir, "v05_three_kernel_metrics.csv")
    tab.to_csv(csv, index=False)

    np.savez_compressed(os.path.join(outdir, "v05_results.npz"),
                        cell_ids=cell_ids,
                        **{k: v for k, v in km.items()
                           if isinstance(v, np.ndarray)})

    with open(os.path.join(outdir, "v05_state.pkl"), "wb") as f:
        pickle.dump({"km": km, "beh": beh, "cell_ids": cell_ids,
                     "adapt_ramp_s_used": km.get("adapt_ramp_s", None),
                     "mod": mod,
                     "cfg_used": {k: getattr(cfg, k) for k in CFG_KEYS
                                  if hasattr(cfg, k)}}, f,
                    protocol=4)

    return csv
