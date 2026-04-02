"""
SMI_Modulation_Evolution.py
Tracks how spatial modulation STRENGTH evolves for the same neurons across sessions.

Three analyses:
  1. SMI trajectory      — mean ± SEM SMI per layer across sessions
  2. Rp vs Rn            — which component (preferred response vs suppression)
                           drives any SMI change
  3. Recruitment/dropout — which cells gain or lose spatial modulation,
                           and whether this differs by layer

Requires: *_smi_results.h5 files (from SMI_FullSession_Batch.py)

JSY, 2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from load_tracked import (
    load_tracking, filter_to_analysis_days, find_smi_files, find_files_from_tracking,
    assign_layers_from_smi, load_smi_session, build_matrix,
    parse_day_numbers, layer_mean_sem, animal_id_from_path,
    LAYER_ORDER, LAYER_COLORS, report_found_files,
)

# ============================================================
# CONFIGURATION
# ============================================================

ROI_TRACKING_FILE = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\roi_tracking_results.h5"
ANIMAL_DIR        = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging"
REFERENCE_DAY     = "Day2"
ANALYSIS_DAYS     = None     # e.g. ['Day2','Day3','Day4','Day5','Day6','Day7']
                             # None = use all tracked sessions

SMI_THRESHOLD     = 0.1          # above = spatially modulated
OUTPUT_DIR        = None         # None -> same folder as this script

# ============================================================


# ── extractors for build_matrix ─────────────────────────────

def _extract_smi(smi_file, roi_indices):
    smi_all, valid, *_ = load_smi_session(smi_file)[:2], *load_smi_session(smi_file)[2:]
    # Re-load cleanly
    smi_all, valid, Rp, Rn, pref_pos, _ = load_smi_session(smi_file)
    out = np.full(len(roi_indices), np.nan)
    for i, roi in enumerate(roi_indices):
        if roi >= 0 and roi < len(smi_all):
            out[i] = smi_all[roi]
    return out


def _extract_Rp(smi_file, roi_indices):
    _, _, Rp, *_ = load_smi_session(smi_file)
    out = np.full(len(roi_indices), np.nan)
    for i, roi in enumerate(roi_indices):
        if roi >= 0 and roi < len(Rp):
            out[i] = Rp[roi]
    return out


def _extract_Rn(smi_file, roi_indices):
    _, _, _, Rn, *_ = load_smi_session(smi_file)
    out = np.full(len(roi_indices), np.nan)
    for i, roi in enumerate(roi_indices):
        if roi >= 0 and roi < len(Rn):
            out[i] = Rn[roi]
    return out


# ── Analysis 1: SMI trajectory ───────────────────────────────

def plot_smi_trajectory(layer_stats, session_days, day_labels, animal_id, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 4))

    x = np.array(session_days)
    for layer in LAYER_ORDER:
        if layer not in layer_stats:
            continue
        ls = layer_stats[layer]
        c  = LAYER_COLORS[layer]
        n  = int(np.nanmax(ls['n']))
        ax.plot(x, ls['mean'], 'o-', color=c, label=f"{layer} (n={n})",
                linewidth=2, markersize=5)
        ax.fill_between(x, ls['mean'] - ls['sem'], ls['mean'] + ls['sem'],
                        color=c, alpha=0.2)

    ax.axhline(SMI_THRESHOLD, color='gray', linestyle=':', alpha=0.6,
               label=f'Threshold ({SMI_THRESHOLD})')
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean SMI ± SEM')
    ax.set_title('SMI Trajectory')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2, axis='y')

    if standalone:
        fig.suptitle(f"{animal_id} — SMI Trajectory by Layer", fontweight='bold')
        plt.tight_layout()
        return fig
    return ax


# ── Analysis 2: Rp vs Rn decomposition ──────────────────────

def plot_rp_rn(rp_stats, rn_stats, session_days, day_labels, animal_id,
               axes=None):
    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    x = np.array(session_days)
    for layer in LAYER_ORDER:
        c = LAYER_COLORS[layer]
        for stat_dict, ax, label in [(rp_stats, axes[0], 'Rp (preferred)'),
                                     (rn_stats, axes[1], 'Rn (non-preferred)')]:
            if layer not in stat_dict:
                continue
            ls = stat_dict[layer]
            ax.plot(x, ls['mean'], 'o-', color=c, label=layer, linewidth=2, markersize=5)
            ax.fill_between(x, ls['mean'] - ls['sem'], ls['mean'] + ls['sem'],
                            color=c, alpha=0.2)

    for ax, title in zip(axes, ['Rp (preferred response)', 'Rn (non-preferred response)']):
        ax.set_xticks(x)
        ax.set_xticklabels(day_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Response amplitude')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    if standalone:
        fig.suptitle(f"{animal_id} — Rp vs Rn Decomposition", fontweight='bold')
        plt.tight_layout()
        return fig
    return axes


# ── Analysis 3: Recruitment / Dropout ───────────────────────

def compute_modulation_state(smi_matrix, threshold=SMI_THRESHOLD):
    """
    Returns boolean matrix: True = spatially modulated (SMI > threshold).
    NaN sessions remain NaN (represented as -1 here for clarity).
    """
    return smi_matrix > threshold   # NaN > threshold = False (numpy behaviour)


def compute_recruitment_dropout(smi_matrix, cell_layers, day_labels,
                                 threshold=SMI_THRESHOLD):
    """
    For each layer, compute per-session:
      - recruitment rate: cells that were NOT modulated in the previous session
                          and ARE modulated now
      - dropout rate:     cells that WERE modulated and are NOT now
      - stable-on rate:   modulated in both consecutive sessions
      - stable-off rate:  not modulated in either consecutive session

    Returns
    -------
    transitions : dict {layer: {'recruit', 'dropout', 'stable_on', 'stable_off'}}
        Each value is an array of length (n_sessions - 1).
    """
    mod = (smi_matrix > threshold)  # (n_cells, n_sessions), NaN->False
    has_data = ~np.isnan(smi_matrix)

    transitions = {}
    for layer in LAYER_ORDER:
        if layer not in cell_layers:
            continue
        rows = cell_layers[layer]
        lmod = mod[rows, :]
        ldata = has_data[rows, :]

        recruit = np.full(smi_matrix.shape[1] - 1, np.nan)
        dropout = np.full(smi_matrix.shape[1] - 1, np.nan)
        stable_on  = np.full(smi_matrix.shape[1] - 1, np.nan)
        stable_off = np.full(smi_matrix.shape[1] - 1, np.nan)

        for s in range(smi_matrix.shape[1] - 1):
            valid = ldata[:, s] & ldata[:, s + 1]
            if np.sum(valid) == 0:
                continue
            prev = lmod[valid, s]
            curr = lmod[valid, s + 1]
            n = np.sum(valid)
            recruit[s]   = np.sum(~prev &  curr) / n * 100
            dropout[s]   = np.sum( prev & ~curr) / n * 100
            stable_on[s] = np.sum( prev &  curr) / n * 100
            stable_off[s]= np.sum(~prev & ~curr) / n * 100

        transitions[layer] = {
            'recruit':    recruit,
            'dropout':    dropout,
            'stable_on':  stable_on,
            'stable_off': stable_off,
        }
    return transitions


def plot_proportion_modulated(smi_matrix, cell_layers, session_days, day_labels,
                               animal_id, threshold=SMI_THRESHOLD, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 4))

    x = np.array(session_days)
    for layer in LAYER_ORDER:
        if layer not in cell_layers:
            continue
        rows = cell_layers[layer]
        data = smi_matrix[rows, :]
        prop = np.nanmean(data > threshold, axis=0) * 100
        n    = int(len(rows))
        ax.plot(x, prop, 's--', color=LAYER_COLORS[layer],
                label=f"{layer} (n={n})", linewidth=1.5, markersize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(f'% Cells with SMI > {threshold}')
    ax.set_title('Proportion Spatially Modulated')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    if standalone:
        fig.suptitle(f"{animal_id} — Proportion Modulated", fontweight='bold')
        plt.tight_layout()
        return fig
    return ax


def plot_recruitment_dropout(transitions, session_days, day_labels, animal_id,
                              output_path=None):
    layers = [l for l in LAYER_ORDER if l in transitions]
    n = len(layers)
    if n == 0:
        return None

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 6), sharey='row', sharex='col')
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f"{animal_id} — Session-to-Session Recruitment & Dropout by Layer",
                 fontweight='bold', fontsize=11)

    # x-axis: between sessions (midpoints)
    x = np.array(session_days)
    x_mid = (x[:-1] + x[1:]) / 2
    x_labels = [f"{day_labels[i]}→{day_labels[i+1]}" for i in range(len(day_labels) - 1)]

    for col, layer in enumerate(layers):
        t = transitions[layer]
        c = LAYER_COLORS[layer]

        # Top row: recruitment & dropout
        ax0 = axes[0, col]
        ax0.plot(x_mid, t['recruit'], 'o-', color='green', label='Recruited', linewidth=2)
        ax0.plot(x_mid, t['dropout'], 's--', color='red',   label='Dropped',   linewidth=2)
        ax0.set_title(layer, fontweight='bold', color=c)
        ax0.set_ylabel('% cells' if col == 0 else '')
        ax0.legend(fontsize=8)
        ax0.set_ylim(0, 100)
        ax0.set_xticks(x_mid)
        ax0.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)
        ax0.grid(True, alpha=0.2, axis='y')

        # Bottom row: stable on/off
        ax1 = axes[1, col]
        ax1.plot(x_mid, t['stable_on'],  'o-', color='steelblue', label='Stable on',  linewidth=2)
        ax1.plot(x_mid, t['stable_off'], 's--', color='lightgray', label='Stable off', linewidth=2)
        ax1.set_ylabel('% cells' if col == 0 else '')
        ax1.legend(fontsize=8)
        ax1.set_ylim(0, 100)
        ax1.set_xticks(x_mid)
        ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)
        ax1.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def plot_delta_smi(smi_matrix, cell_layers, day_labels, session_days,
                   animal_id, output_path=None):
    """Per-cell ΔSMI (last 2 sessions − first 2 sessions), violin by layer."""
    n_sessions = smi_matrix.shape[1]
    early_idx = list(range(min(2, n_sessions)))
    late_idx  = list(range(max(0, n_sessions - 2), n_sessions))
    early_lbl = day_labels[early_idx[0]]
    late_lbl  = day_labels[late_idx[-1]]

    layers = [l for l in LAYER_ORDER if l in cell_layers and len(cell_layers[l]) > 0]
    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 5), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    fig.suptitle(f"{animal_id} — ΔSMI ({late_lbl} − {early_lbl}) by Layer",
                 fontweight='bold')

    for ax, layer in zip(axes, layers):
        rows = cell_layers[layer]
        data = smi_matrix[rows, :]
        early = np.nanmean(data[:, early_idx], axis=1)
        late  = np.nanmean(data[:, late_idx],  axis=1)
        delta = late - early
        valid = ~np.isnan(early) & ~np.isnan(late)
        delta = delta[valid]

        c = LAYER_COLORS[layer]
        parts = ax.violinplot([delta], positions=[0], showmedians=True, showextrema=False)
        parts['bodies'][0].set_facecolor(c); parts['bodies'][0].set_alpha(0.5)
        parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(delta))
        ax.scatter(jitter, delta, color=c, alpha=0.4, s=15, zorder=3)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.7)

        # Wilcoxon vs 0
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
        ax.set_ylabel('ΔSMI' if ax is axes[0] else '')
        ax.set_xticks([])
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def run_statistics(smi_matrix, cell_layers, day_labels):
    """
    Per-layer statistics using pairwise Wilcoxon (first session vs each other),
    comparing only cells with valid SMI in BOTH sessions being compared.
    This avoids the near-zero 'n complete' problem that comes from requiring
    valid data across all sessions simultaneously.
    """
    print("\n" + "=" * 65)
    print("STATISTICS — SMI change across sessions")
    print("  (pairwise Wilcoxon: each session vs first, paired on valid cells)")
    print("=" * 65)
    ref_col = 0   # first session = reference

    for layer in LAYER_ORDER:
        if layer not in cell_layers:
            continue
        rows = cell_layers[layer]
        data = smi_matrix[rows, :]
        ref  = data[:, ref_col]

        print(f"\n{layer}")
        print(f"  {'Session':<10} {'n_pairs':>7}  {'med ref':>8}  {'med sess':>9}  {'p':>8}  sig")
        print(f"  {'-'*55}")

        for col in range(1, data.shape[1]):
            sess = data[:, col]
            valid = ~np.isnan(ref) & ~np.isnan(sess)
            n = np.sum(valid)
            if n < 5:
                print(f"  {day_labels[col]:<10} {n:>7}  {'–':>8}  {'–':>9}  {'n<5':>8}")
                continue
            try:
                _, p = stats.wilcoxon(ref[valid], sess[valid])
            except Exception:
                _, p = stats.ttest_rel(ref[valid], sess[valid])
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            med_ref  = np.median(ref[valid])
            med_sess = np.median(sess[valid])
            direction = "↑" if med_sess > med_ref else "↓"
            print(f"  {day_labels[col]:<10} {n:>7}  {med_ref:>8.3f}  "
                  f"{med_sess:>8.3f}{direction}  {p:>8.4f}  {sig}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("SMI MODULATION EVOLUTION — TRACKED ROIs")
    print("=" * 65)

    animal_id = animal_id_from_path(ROI_TRACKING_FILE)
    out_dir = OUTPUT_DIR or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"{animal_id}_SMI_Evolution")

    # Load
    print("\n[1] Loading tracking matrix...")
    tracked_matrix, day_labels, session_dirs = load_tracking(ROI_TRACKING_FILE)
    tracked_matrix, day_labels, session_dirs = filter_to_analysis_days(
        tracked_matrix, day_labels, session_dirs, ANALYSIS_DAYS)
    session_days = parse_day_numbers(day_labels)

    print("\n[2] Finding SMI result files...")
    if session_dirs:
        smi_files = find_files_from_tracking(session_dirs, "*_smi_results.h5")
    else:
        smi_files = find_smi_files(ANIMAL_DIR)
    report_found_files("SMI files", smi_files, day_labels)

    # Build matrices
    print("\n[3] Building SMI / Rp / Rn matrices...")
    smi_matrix = build_matrix(tracked_matrix, day_labels, smi_files, _extract_smi)
    rp_matrix  = build_matrix(tracked_matrix, day_labels, smi_files, _extract_Rp)
    rn_matrix  = build_matrix(tracked_matrix, day_labels, smi_files, _extract_Rn)

    n_valid = np.sum(~np.isnan(smi_matrix), axis=0)
    for d, nv in zip(day_labels, n_valid):
        print(f"  {d}: {nv}/{tracked_matrix.shape[0]} tracked cells with valid SMI")

    # Layer assignment
    print("\n[4] Assigning cells to layers...")
    cell_layers = assign_layers_from_smi(tracked_matrix, day_labels, smi_files, REFERENCE_DAY)
    for l in LAYER_ORDER:
        print(f"  {l}: {len(cell_layers.get(l, []))} tracked cells")

    # Stats
    run_statistics(smi_matrix, cell_layers, day_labels)

    # Figures
    print("\n[5] Generating figures...")

    smi_stats = layer_mean_sem(smi_matrix, cell_layers)
    rp_stats  = layer_mean_sem(rp_matrix,  cell_layers)
    rn_stats  = layer_mean_sem(rn_matrix,  cell_layers)
    transitions = compute_recruitment_dropout(smi_matrix, cell_layers, day_labels)

    # Combined overview figure (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{animal_id} — Spatial Modulation Strength Evolution",
                 fontweight='bold', fontsize=13)

    plot_smi_trajectory(smi_stats, session_days, day_labels, animal_id, ax=axes[0, 0])
    plot_proportion_modulated(smi_matrix, cell_layers, session_days, day_labels,
                               animal_id, ax=axes[0, 1])
    plot_rp_rn(rp_stats, rn_stats, session_days, day_labels, animal_id,
               axes=[axes[1, 0], axes[1, 1]])

    plt.tight_layout()
    overview_path = f"{prefix}_overview.png"
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {os.path.basename(overview_path)}")

    # Delta SMI
    plot_delta_smi(smi_matrix, cell_layers, day_labels, session_days, animal_id,
                   output_path=f"{prefix}_deltaSMI.png")

    # Recruitment / dropout
    plot_recruitment_dropout(transitions, session_days, day_labels, animal_id,
                              output_path=f"{prefix}_recruitment_dropout.png")

    plt.show()
    print(f"\nDone. Outputs in: {out_dir}")

    return dict(tracked_matrix=tracked_matrix, smi_matrix=smi_matrix,
                rp_matrix=rp_matrix, rn_matrix=rn_matrix,
                cell_layers=cell_layers, day_labels=day_labels,
                session_days=session_days)


if __name__ == "__main__":
    main()
