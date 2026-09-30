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


def plot_summary(res, cfg, path):

    lab = res["label"]
    landmark = np.isin(lab, LABELS[:4])
    visual = lab == "visual"
    use = landmark | visual
    fig, axs = plt.subplots(3, 3, figsize=(10, 9.5))
    axs = axs.ravel()

    def _hist(ax, v, bins, xlabel, thr=None):
        ax.hist([v[landmark & np.isfinite(v)], v[visual & np.isfinite(v)]],
                bins=bins, stacked=True, color=[C_FREE, C_TIED],
                edgecolor="w", lw=0.5, label=["landmark", "visual"])
        if thr is not None:
            ax.axvline(thr, color=C_DATA, lw=0.8, ls="--")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("cells", fontsize=8)
        _style(ax)

    _hist(axs[0], res["psi_2nd"], np.linspace(0, 1, 21), "PSI")
    axs[0].legend(fontsize=7, frameon=False)
    _hist(axs[1], res["n_eff"], np.linspace(1, 4, 25), "n_eff")
    _hist(axs[2], res["dr2_cv"], 25, "FFI (held-out R2, free - equal fit)",
          cfg.gauss_visual_max_dr2)

    ax = axs[3]
    for name, m, col in [("landmark", landmark, C_FREE), ("visual", visual, C_TIED)]:
        ax.scatter(res["n_eff"][m], res["dr2_cv"][m], s=12, color=col,
                   edgecolor="w", lw=0.5, label="{} ({})".format(name, int(m.sum())))
    ax.axhline(0, color=C_MUTED, lw=0.8)
    ax.set_xlabel("n_eff", fontsize=8)
    ax.axhline(cfg.gauss_visual_max_dr2, color=C_DATA, lw=0.8, ls="--")
    ax.set_ylabel("FFI (held-out R2, free - equal fit)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)

    ax = axs[4]
    for m, col in [(landmark, C_FREE), (visual, C_TIED)]:
        ax.scatter(res["psi_2nd"][m], res["n_eff"][m], s=12, color=col,
                   edgecolor="w", lw=0.5)
    ax.set_xlabel("PSI", fontsize=8)
    ax.set_ylabel("n_eff", fontsize=8)
    _style(ax)

    ax = axs[5]
    counts = [int(np.sum(lab == k)) for k in LABELS]
    ax.bar(np.arange(len(LABELS)), counts, color=C_FREE, width=0.6)
    for i, n in enumerate(counts):
        ax.text(i, n, str(n), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(LABELS)), LABELS, fontsize=7, rotation=30)
    ax.set_ylabel("cells", fontsize=8)
    _style(ax)

    _hist(axs[6], res["delta"], np.linspace(-cfg.gauss_delta_max_cm,
                                            cfg.gauss_delta_max_cm, 25),
          "delta (cm)")
    _hist(axs[7], res["sigma"], np.linspace(cfg.gauss_sigma_min_cm,
                                            cfg.gauss_sigma_max_cm, 22),
          "sigma (cm)")

    ax = axs[8]
    ax.hist(res["rel_null"][np.isfinite(res["rel_null"])], bins=40,
            color=C_MUTED, alpha=0.8, label="shift null", density=True)
    ax.hist(res["rel_r"][np.isfinite(res["rel_r"])], bins=40, color=C_FREE,
            alpha=0.6, label="real", density=True)
    ax.axvline(max(res["rel_thr"], cfg.gauss_rel_min_r), color=C_DATA, lw=0.8,
               ls="--")
    ax.set_xlabel("reliability (split-half r)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
