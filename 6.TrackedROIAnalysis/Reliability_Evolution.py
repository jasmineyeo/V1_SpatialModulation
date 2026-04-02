"""
Reliability_Evolution.py
Tracks whether spatial responses become more consistent (reliable) across sessions
for the same neurons — separate from the question of *how selective* they are (SMI).

Two metrics from preproc.h5:
  - avg_cc   : mean Pearson cross-correlation between odd/even trial responses
               (trial-to-trial consistency of the spatial tuning curve)
  - cohen_d  : effect size separating active vs inactive spatial bins
               (signal-to-noise of the spatial response)

Both are computed during preprocessing and stored per cell.

Analyses:
  1. Reliability trajectory  — mean ± SEM avg_cc and cohen_d per layer
  2. Δ Reliability           — per-cell change (late − early), violin by layer
  3. SMI vs Reliability      — scatter: does reliability predict SMI trajectory?
                               (requires *_smi_results.h5 as well)

Requires:
  - *_preproc*.h5       (avg_cc, cohen_d, combined_reliable)
  - *_smi_results.h5    (optional, for SMI vs reliability scatter)

JSY, 2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from load_tracked import (
    load_tracking, filter_to_analysis_days, find_smi_files, find_preproc_files,
    find_files_from_tracking, assign_layers_from_smi, load_smi_session,
    load_preproc_session, build_matrix, parse_day_numbers, layer_mean_sem,
    animal_id_from_path, LAYER_ORDER, LAYER_COLORS, report_found_files,
)

# ============================================================
# CONFIGURATION
# ============================================================

ROI_TRACKING_FILE = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\roi_tracking_JSY054.h5"
ANIMAL_DIR        = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging"
REFERENCE_DAY     = "Day2"
ANALYSIS_DAYS     = None     # e.g. ['Day2','Day3','Day4','Day5','Day6','Day7']
                             # None = use all tracked sessions
OUTPUT_DIR        = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\TrackingResults"   # None -> same folder as this script

# ============================================================


# ── Extractors ───────────────────────────────────────────────

def _extract_avg_cc(preproc_file, roi_indices):
    _, _, _, _, cc, _, _ = load_preproc_session(preproc_file)
    out = np.full(len(roi_indices), np.nan)
    for i, roi in enumerate(roi_indices):
        if roi >= 0 and roi < len(cc):
            out[i] = cc[roi]
    return out


def _extract_cohen_d(preproc_file, roi_indices):
    _, _, _, _, _, cd, _ = load_preproc_session(preproc_file)
    out = np.full(len(roi_indices), np.nan)
    for i, roi in enumerate(roi_indices):
        if roi >= 0 and roi < len(cd):
            out[i] = cd[roi]
    return out


def _extract_smi_for_scatter(smi_file, roi_indices):
    smi_all, valid, *_ = load_smi_session(smi_file)[:2], *load_smi_session(smi_file)[2:]
    smi_all, valid, *_ = load_smi_session(smi_file)
    out = np.full(len(roi_indices), np.nan)
    for i, roi in enumerate(roi_indices):
        if roi >= 0 and roi < len(smi_all):
            out[i] = smi_all[roi]
    return out


# ── Analysis 1: Reliability trajectory ──────────────────────

def plot_reliability_trajectory(cc_stats, cd_stats, session_days, day_labels,
                                  animal_id, output_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"{animal_id} — Reliability Evolution", fontweight='bold')

    x = np.array(session_days)

    for ax, stat_dict, ylabel, title in [
        (axes[0], cc_stats,  'Mean avg_cc ± SEM',    'Trial-to-Trial Consistency (avg_cc)'),
        (axes[1], cd_stats,  "Mean Cohen's d ± SEM", "Response SNR (Cohen's d)"),
    ]:
        for layer in LAYER_ORDER:
            if layer not in stat_dict:
                continue
            ls = stat_dict[layer]
            c  = LAYER_COLORS[layer]
            n  = int(np.nanmax(ls['n']))
            ax.plot(x, ls['mean'], 'o-', color=c, label=f"{layer} (n={n})",
                    linewidth=2, markersize=5)
            ax.fill_between(x, ls['mean'] - ls['sem'], ls['mean'] + ls['sem'],
                            color=c, alpha=0.2)
        ax.set_xlabel('Recording Day')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(day_labels, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


# ── Analysis 2: Δ Reliability ────────────────────────────────

def plot_delta_reliability(matrix, cell_layers, day_labels, metric_name,
                            animal_id, output_path=None):
    """Per-cell change (last 2 − first 2 sessions), violin by layer."""
    n_sessions = matrix.shape[1]
    early_idx  = list(range(min(2, n_sessions)))
    late_idx   = list(range(max(0, n_sessions - 2), n_sessions))
    early_lbl  = day_labels[early_idx[0]]
    late_lbl   = day_labels[late_idx[-1]]

    layers = [l for l in LAYER_ORDER if l in cell_layers and len(cell_layers[l]) > 0]
    if not layers:
        return None

    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 5), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    fig.suptitle(
        f"{animal_id} — Δ{metric_name} ({late_lbl} − {early_lbl}) by Layer",
        fontweight='bold')

    for ax, layer in zip(axes, layers):
        rows  = cell_layers[layer]
        data  = matrix[rows, :]
        early = np.nanmean(data[:, early_idx], axis=1)
        late  = np.nanmean(data[:, late_idx],  axis=1)
        delta = late - early
        valid = ~np.isnan(early) & ~np.isnan(late)
        delta = delta[valid]
        c     = LAYER_COLORS[layer]

        parts = ax.violinplot([delta], positions=[0], showmedians=True, showextrema=False)
        parts['bodies'][0].set_facecolor(c); parts['bodies'][0].set_alpha(0.5)
        parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(delta))
        ax.scatter(jitter, delta, color=c, alpha=0.4, s=15, zorder=3)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)

        p_str, sig = '', 'ns'
        if len(delta) >= 5:
            try:
                _, p = stats.wilcoxon(delta)
            except Exception:
                _, p = stats.ttest_1samp(delta, 0)
            p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
            sig   = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")

        med = np.nanmedian(delta)
        ax.set_title(f"{layer}\nn={len(delta)}, med={med:+.3f}\n{sig} {p_str}", fontsize=9)
        ax.set_ylabel(f"Δ{metric_name}" if ax is axes[0] else '')
        ax.set_xticks([])
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


# ── Analysis 3: SMI vs Reliability scatter ──────────────────

def plot_smi_vs_reliability(smi_matrix, cc_matrix, cell_layers,
                              session_days, day_labels, animal_id,
                              output_path=None):
    """
    For each layer: scatter of SMI vs avg_cc, coloured by session.
    Also shows Spearman correlation per layer.
    """
    layers = [l for l in LAYER_ORDER if l in cell_layers and len(cell_layers[l]) > 0]
    n_layers = len(layers)
    if n_layers == 0:
        return None

    cmap = plt.cm.viridis
    n_sessions = smi_matrix.shape[1]
    colors_session = [cmap(i / max(n_sessions - 1, 1)) for i in range(n_sessions)]

    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4.5))
    if n_layers == 1:
        axes = [axes]
    fig.suptitle(f"{animal_id} — SMI vs avg_cc by Layer", fontweight='bold')

    for ax, layer in zip(axes, layers):
        rows = cell_layers[layer]
        all_smi = []
        all_cc  = []
        all_col = []

        for col in range(n_sessions):
            smi_vals = smi_matrix[rows, col]
            cc_vals  = cc_matrix[rows, col]
            valid = ~np.isnan(smi_vals) & ~np.isnan(cc_vals)
            all_smi.extend(smi_vals[valid].tolist())
            all_cc.extend(cc_vals[valid].tolist())
            all_col.extend([colors_session[col]] * int(np.sum(valid)))
            ax.scatter(cc_vals[valid], smi_vals[valid],
                       color=colors_session[col], alpha=0.35, s=12, zorder=2)

        # Spearman over all data pooled
        if len(all_smi) >= 10:
            rho, p = stats.spearmanr(all_cc, all_smi)
            p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
            ax.text(0.05, 0.95, f"ρ={rho:.2f}\n{p_str}",
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('avg_cc (reliability)')
        ax.set_ylabel('SMI' if ax is axes[0] else '')
        ax.set_title(layer, fontweight='bold', color=LAYER_COLORS[layer])
        ax.grid(True, alpha=0.2)

    # Colorbar for session
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=session_days[0],
                                                   vmax=session_days[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[-1], fraction=0.04, pad=0.04)
    cbar.set_label('Recording Day')
    tick_pos = np.linspace(session_days[0], session_days[-1], min(5, n_sessions))
    cbar.set_ticks(tick_pos)
    cbar.set_ticklabels([str(int(t)) for t in tick_pos])

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def run_reliability_statistics(cc_matrix, cd_matrix, cell_layers, day_labels):
    """Friedman test per layer for each reliability metric."""
    print("\n" + "=" * 65)
    print("STATISTICS — Reliability change across sessions")
    print("=" * 65)

    for metric_name, matrix in [("avg_cc", cc_matrix), ("Cohen's d", cd_matrix)]:
        print(f"\n── {metric_name} ──")
        for layer in LAYER_ORDER:
            if layer not in cell_layers:
                continue
            rows = cell_layers[layer]
            data = matrix[rows, :]
            complete = ~np.any(np.isnan(data), axis=1)
            n_c = np.sum(complete)

            print(f"\n  {layer}  (n complete={n_c})")
            if n_c < 3:
                first = data[:, 0]; last = data[:, -1]
                v = ~np.isnan(first) & ~np.isnan(last)
                if np.sum(v) >= 5:
                    try:
                        _, p = stats.wilcoxon(first[v], last[v])
                        print(f"    First vs Last Wilcoxon: p={p:.4f} | "
                              f"med {np.nanmedian(first[v]):.3f}→{np.nanmedian(last[v]):.3f}")
                    except Exception:
                        pass
                continue

            cd = data[complete, :]
            try:
                chi2, p_f = stats.friedmanchisquare(*[cd[:, i] for i in range(cd.shape[1])])
                sig = "**" if p_f < 0.01 else ("*" if p_f < 0.05 else "ns")
                print(f"    Friedman: chi2={chi2:.2f}, p={p_f:.4f} {sig}")
                if p_f < 0.05:
                    for col in range(1, cd.shape[1]):
                        try:
                            _, p_w = stats.wilcoxon(cd[:, 0], cd[:, col])
                            s = "**" if p_w < 0.01 else ("*" if p_w < 0.05 else "ns")
                            print(f"    {day_labels[0]} vs {day_labels[col]}: "
                                  f"p={p_w:.4f} {s}  "
                                  f"med {np.median(cd[:,0]):.3f}→{np.median(cd[:,col]):.3f}")
                        except Exception:
                            pass
            except Exception as e:
                print(f"    Friedman failed: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("RELIABILITY EVOLUTION — TRACKED ROIs")
    print("=" * 65)

    animal_id = animal_id_from_path(ROI_TRACKING_FILE)
    out_dir   = OUTPUT_DIR or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"{animal_id}_Reliability")

    # Load
    print("\n[1] Loading tracking matrix...")
    tracked_matrix, day_labels, session_dirs = load_tracking(ROI_TRACKING_FILE)
    tracked_matrix, day_labels, session_dirs = filter_to_analysis_days(
        tracked_matrix, day_labels, session_dirs, ANALYSIS_DAYS)
    session_days = parse_day_numbers(day_labels)

    print("\n[2] Finding data files...")
    if session_dirs:
        smi_files     = find_files_from_tracking(session_dirs, "*_smi_results.h5")
        preproc_files = find_files_from_tracking(session_dirs, "*_preproc*.h5")
    else:
        smi_files     = find_smi_files(ANIMAL_DIR)
        preproc_files = find_preproc_files(ANIMAL_DIR)
    report_found_files("SMI files",     smi_files,     day_labels)
    report_found_files("Preproc files", preproc_files, day_labels)

    # Layer assignment (from SMI files)
    print("\n[3] Assigning cells to layers...")
    cell_layers = assign_layers_from_smi(tracked_matrix, day_labels,
                                          smi_files, REFERENCE_DAY)
    for l in LAYER_ORDER:
        print(f"  {l}: {len(cell_layers.get(l, []))} tracked cells")

    # Build reliability matrices
    print("\n[4] Building avg_cc and Cohen's d matrices...")
    cc_matrix = build_matrix(tracked_matrix, day_labels, preproc_files, _extract_avg_cc)
    cd_matrix = build_matrix(tracked_matrix, day_labels, preproc_files, _extract_cohen_d)

    n_valid_cc = np.sum(~np.isnan(cc_matrix), axis=0)
    for d, nv in zip(day_labels, n_valid_cc):
        print(f"  {d}: {nv}/{tracked_matrix.shape[0]} cells with avg_cc")

    # Stats
    run_reliability_statistics(cc_matrix, cd_matrix, cell_layers, day_labels)

    # Figures
    print("\n[5] Generating figures...")
    cc_stats = layer_mean_sem(cc_matrix, cell_layers)
    cd_stats = layer_mean_sem(cd_matrix, cell_layers)

    # Trajectory
    plot_reliability_trajectory(cc_stats, cd_stats, session_days, day_labels,
                                  animal_id, output_path=f"{prefix}_trajectory.png")

    # Delta avg_cc
    plot_delta_reliability(cc_matrix, cell_layers, day_labels, "avg_cc",
                            animal_id, output_path=f"{prefix}_delta_avgcc.png")

    # Delta Cohen's d
    plot_delta_reliability(cd_matrix, cell_layers, day_labels, "Cohen_d",
                            animal_id, output_path=f"{prefix}_delta_cohend.png")

    # SMI vs reliability scatter (if SMI files available)
    if smi_files:
        smi_matrix = build_matrix(tracked_matrix, day_labels, smi_files,
                                   _extract_smi_for_scatter)
        plot_smi_vs_reliability(smi_matrix, cc_matrix, cell_layers,
                                 session_days, day_labels, animal_id,
                                 output_path=f"{prefix}_SMI_vs_reliability.png")

    plt.show()
    print(f"\nDone. Outputs in: {out_dir}")

    return dict(tracked_matrix=tracked_matrix, cc_matrix=cc_matrix,
                cd_matrix=cd_matrix, cell_layers=cell_layers,
                day_labels=day_labels, session_days=session_days)


if __name__ == "__main__":
    main()
