# -*- coding: utf-8 -*-
"""
landmarkGauss/plots_gauss.py

DMM, Sept 2026
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from fit_gauss import gauss_model


LABELS = ["L1-pref", "L2-pref", "L3-pref", "L4-pref", "visual"]

C_DATA = "#52514e"
C_FREE = "#2a78d6"
C_TIED = "#eb6834"
C_MUTED = "#b7b6b0"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)


ROWS = 5
COLS = 2


def _cell_panel(ax, c, res, disp, cfg, cell_id, legend=False):
    
    lo, hi = res["span"]
    xd, Td, Sd = disp["x"], disp["T"][:, c], disp["sem"][:, c]

    for zl, zh in cfg.zones:
        ax.axvspan(zl, zh, color="#f2c14e", alpha=.25, lw=0)
    ax.axvline(hi, color="#1f4e79", lw=.8, ls=":")
    ax.axvline(lo, color="#1f4e79", lw=.8, ls=":")
    ax.axvline(cfg.reward_cm, color="#7d3c98", lw=.8, ls="--")

    ax.fill_between(xd, Td - Sd, Td + Sd, color="0.15", alpha=.30, lw=0)
    ax.plot(xd, Td, color="0.15", lw=1.3)

    xf = np.linspace(lo, hi, 400)
    cen = res["centers"]
    ax.plot(xf, gauss_model(xf, cen, res["delta_tied"][c], res["sigma_tied"][c],
                            np.full(4, res["A_tied"][c]), res["b_tied"][c]),
            color=C_TIED, lw=1.3, alpha=.75, label="equal-fit")
    ax.plot(xf, gauss_model(xf, cen, res["delta"][c], res["sigma"][c],
                            res["A"][:, c], res["b"][c]),
            color=C_FREE, lw=1.3, alpha=.75, label="free-fit")

    top = np.nanmax(Td + Sd)
    ax.set_ylim(0.0, float(top) * 1.10 if np.isfinite(top) and top > 0 else 1.0)
    ax.set_xlim(0.0, max(cfg.corridor_cm, cfg.reward_cm) + 1.5)
    ax.tick_params(labelsize=5, length=2)
    ax.set_xlabel("position (cm)", fontsize=6)
    ax.set_ylabel("rate", fontsize=6)

    ax.set_title("cell {}, {}, r={:.2f}, FFI={:.3f}".format(
        int(cell_id), res["label"][c], res["rel_r"][c], res["dr2_cv"][c]),
        fontsize=6.5, color="k", loc="left")
    if legend:
        ax.legend(fontsize=5.5, frameon=False, loc="upper right",
                  handlelength=1.5)


def plot_cells_pdf(res, disp, cfg, cell_ids, cells, path):

    per = ROWS * COLS
    with PdfPages(path) as pdf:
        for p0 in range(0, max(len(cells), 1), per):
            sel = cells[p0:p0 + per]
            fig, axs = plt.subplots(ROWS, COLS, figsize=(8.5, 11.0),
                                    squeeze=False)
            fig.subplots_adjust(left=.07, right=.98, top=.95, bottom=.04,
                                hspace=.55, wspace=.18)
            for k, ax in enumerate(axs.ravel()):
                if k >= len(sel):
                    ax.set_visible(False)
                    continue
                _cell_panel(ax, sel[k], res, disp, cfg, cell_ids[sel[k]],
                            legend=k == 0)
            pdf.savefig(fig)
            plt.close(fig)
