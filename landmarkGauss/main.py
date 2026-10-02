# -*- coding: utf-8 -*-
"""
landmarkGauss/main.py

Do the fit on the lap avg, not on the full continuous recording (unlike the
GLM). gaussian fit uses

    f(x) = b + sum_k [ A_k * exp(-(x-(c_k + delta))^2 / (2 sigma^2)) ]

    and here by sum_k i mean Sigma_{k} ...everything here....
    could just write whole thin in latex notation later

where c_k is the set of landmark midpoints w/ a fxed spacing. delta is the
shared shift, sigma is the shared width, they have a shared baseline and A_k
is the per-landmark amplitudes. and i cal |delta| at half landmark spacing so
the first gausian always belongs to the first landmark (i.e., it does not over-roll).

fits with NNLS. fits that free-amplitude model and then it also fits a fixed-
amplitude model (WHAT CAN I CALL THESE INSTEAD OF FIXED AND FREE... SOMETHING
WHERE THE ACRONYM FOR AN INDEX WOULDN'T BE CONFUSING? FREE AND RIGID?) then i
compare the fixed vs free model's on held-out laps. that's the visual vs 
visual+spatial test. threshold for this is set in gauss_config.py

this does two fits for each of the two models: first a grid fit (e.g., delta
can be -12 to +12 cm in 0.5 cm steps). then fine-tune by starting at the winner
on the grid, and nudge the value around as a continuous value to optimize. with
least-squares. (i'm using scipy.optimize module). this is much more efficient than
treating the entire spread as a continuous search -- find region, then jitter
around it to get to the final solution.

DMM, Sept 2026
"""

import os
import sys
import argparse
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "landmarkGLM"))

from load_data import load_suite2p, load_vrlog
from pure_behavior_block import build_behavior
from gauss_config import GaussConfig
from fit_gauss import fit_all, label_cells
from plots_gauss import LABELS, plot_cells_pdf, plot_summary
from gui_funcs import select_directory, select_file


def main(SUITE2P, VRLOG, OUTDIR):

    os.makedirs(OUTDIR, exist_ok=True)
    cfg = GaussConfig()

    print("loading suite2p and VR log...")
    dff, spks, cell_ids = load_suite2p(SUITE2P, cfg)
    vr_df = load_vrlog(VRLOG)
    beh = build_behavior(vr_df, dff.shape[1], cfg)

    # same normalization as landmarkGLM (mean rate on run frames = 1)
    Y = np.maximum((dff if cfg.zone_target == "dff" else spks).T, 0.0)
    if cfg.glm_rate_norm:
        m = Y[beh["run_mask"]].mean(axis=0)
        Y = Y / np.where(m <= 0, 1.0, m) * float(cfg.glm_mean_rate)

    print("fitting four gaussians...")
    res = fit_all(Y, beh, cfg)

    res = label_cells(res, cfg)
    disp = dict(x=res["disp_x"], T=res["disp_T"], sem=res["disp_sem"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cols = [
        "label", "reliable", "good", "rel_r", "r2_free", "r2_tied",
        "r2cv_free", "r2cv_tied", "dr2_cv", "pref", "pref_consistency",
        "psi_2nd", "psi_min", "psi_mean", "psi_2nd_half_mean",
        "psi_2nd_half_sd", "n_eff", "sparseness", "amp_reward",
        "amp_onset", "delta_at_bound", "delta", "sigma", "b",
        "A_tied", "b_tied"
    ]
    df = pd.DataFrame({k: res[k] for k in cols})
    for k in range(4):
        df.insert(df.columns.get_loc("b"), "A{}".format(k + 1), res["A"][k])
        df.insert(df.columns.get_loc("b"), "A{}_eff".format(k + 1),
                  res["A_eff"][k])
    df.insert(0, "cell_id", cell_ids)
    csv = os.path.join(OUTDIR, "gauss_cells_{}.csv".format(stamp))
    df.to_csv(csv, index=False)

    np.savez(os.path.join(OUTDIR, "gauss_fit_{}.npz".format(stamp)),
             cell_ids=cell_ids,
             **{k: v for k, v in res.items() if k != "label"},
             label=res["label"].astype(str))

    lab = res["label"]
    print("  {} cells: {} reliable, {} good fits -> {}".format(
        len(lab), int(res["reliable"].sum()), int(res["good"].sum()),
        ", ".join("{} {}".format(k, int(np.sum(lab == k))) for k in
                  LABELS)))

    print(" -> Plotting...")
    order = {k: i for i, k in enumerate(LABELS)}
    good = np.flatnonzero(res["good"])
    good = good[np.lexsort((-np.nan_to_num(res["rel_r"][good]),
                            [order[lab[c]] for c in good]))]
    plot_cells_pdf(res, disp, cfg, cell_ids, good,
                   os.path.join(OUTDIR, "gauss_good_cells_{}.pdf".format(stamp)))
    print("  wrote {}".format(csv))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s2p', '--suite2p_dir', type=str, default=None,
                        help='Path to suite2p directory')
    parser.add_argument('-vr', '--vrlog_file', type=str, default=None,
                        help='Path to VR log file')
    parser.add_argument('-b', '--batch', action='store_true')
    parser.add_argument('-bd', '--batch_dir', type=str, default=None)
    args = parser.parse_args()

    if not args.batch:

        if args.suite2p_dir is None:
            SUITE2P = select_directory('Select suite2p plane directory.')
        else:
            SUITE2P = args.suite2p_dir
        if args.vrlog_file is None:
            VRLOG = select_file('Select VR log text file.', filetypes=[('TXT', '*.txt'),])
        else:
            VRLOG = args.vrlog_file

        outdir = os.path.split(VRLOG)[0]
        if outdir == '':
            outdir = os.getcwd()

        main(SUITE2P, VRLOG, outdir)

    elif args.batch:

        bdir = args.batch_dir
        logfiles = glob(os.path.join(bdir, '*/*.txt'), recursive=True)
        for i, logfile in enumerate(logfiles):
            logbase = os.path.split(logfile)[0]
            suite2p_dir = os.path.join(logbase, 'suite2p/plane0')

            main(suite2p_dir, logfile, logbase)

# python landmarkGauss/main.py --batch --batch_dir /home/dylan/Fast1/jasmine_glm/JSY054

# python landmarkGauss/main.py -s2p /home/dylan/Fast1/jasmine_glm/JSY054/251105_JSY_JSY054_SpMod_Day7/suite2p/plane0 -vr /home/dylan/Fast1/jasmine_glm/JSY054/251105_JSY_JSY054_SpMod_Day7/VRlog_JSY054_11052025_03-36-34_forSharing.txt
