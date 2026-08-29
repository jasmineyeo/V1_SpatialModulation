"""
AxonMerge_batch.py
Batch version of AxonMerge_Suite2p.ipynb.

For each suite2p/plane0 directory:
  1. Load F, Fneu, iscell, stat, ops
  2. Compute dF/F
  3. Compute pairwise correlations + distances
  4. Build edges and find connected components (axon groups)
  5. Average traces within groups → dFF_merged
  6. Denoise + infer spikes
  7. Save figures and update iscell.npy

JSY, 2026
"""

import os
import sys
import itertools
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for batch
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
from collections import defaultdict

sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")
from helper.twop import TwoP


# ============================================================================
# CONFIGURATION — edit these paths and parameters
# ============================================================================

SUITE2P_PATHS = [
    r"D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260202_JSY_JSY061_SpMod_AxonalImaging_Day1\TSeries-02022026-1804-001\suite2p\plane0",
    r"D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260203_JSY_JSY061_SpMod_AxonalImaging_Day2\TSeries-02032026-1751-002\suite2p\plane0",
    r"D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260204_JSY_JSY061_SpMod_AxonalImaging_Day3\TSeries-02042026-2009-001\suite2p\plane0",
    r"D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260205_JSY_JSY061_SpMod_AxonalImaging_Day4\TSeries-02052026-1833-002\suite2p\plane0",
    r"D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260206_JSY_JSY061_SpMod_AxonalImaging_Day5\TSeries-02062026-1850-001\suite2p\plane0",
    r"D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260207_JSY_JSY061_SpMod_AxonalImaging_Day6\TSeries-02072026-2023-001\suite2p\plane0",
    r"D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260208_JSY_JSY061_SpMod_AxonalImaging_Day7\TSeries-02082026-1826-001\suite2p\plane0",
]

# Merging parameters (same defaults as notebook)
TWOP_RATE       = 10.047   # imaging frame rate (Hz)
NEU_CORRECTION  = 0        # neuropil correction factor
CC_THRESH       = 0.5      # pairwise correlation threshold
MAX_DISTANCE    = 30       # max centroid distance (px) — only used when USE_MAX_DISTANCE=True
USE_MAX_DISTANCE = True    # True → local edges (cc AND dist)
MERGE_DUPLICATES = True    # True = average correlated groups

# Set to True to skip sessions that have already been merged (iscell_premerge.npy exists)
SKIP_IF_DONE = False


# ============================================================================
# CORE FUNCTION
# ============================================================================

def run_axon_merge(suite2p_path,
                   twop_rate=TWOP_RATE,
                   neu_correction=NEU_CORRECTION,
                   cc_thresh=CC_THRESH,
                   max_distance=MAX_DISTANCE,
                   use_max_distance=USE_MAX_DISTANCE):
    """Run axon merging for a single suite2p/plane0 directory."""

    print(f"\n{'='*80}")
    print(f"  {suite2p_path}")
    print(f"{'='*80}")

    # ── Load suite2p files ────────────────────────────────────────────────
    F      = np.load(os.path.join(suite2p_path, 'F.npy'))
    Fneu   = np.load(os.path.join(suite2p_path, 'Fneu.npy'))
    iscell = np.load(os.path.join(suite2p_path, 'iscell.npy'))
    stat   = np.load(os.path.join(suite2p_path, 'stat.npy'), allow_pickle=True)
    ops    = np.load(os.path.join(suite2p_path, 'ops.npy'),  allow_pickle=True).item()

    cell_mask       = iscell[:, 0] == 1
    F_cells         = F[cell_mask, :]
    Fneu_cells      = Fneu[cell_mask, :]
    stat_cells      = stat[cell_mask]
    original_indices = np.where(cell_mask)[0]

    print(f"Total ROIs: {len(stat)}  |  after iscell filter: {F_cells.shape[0]}  |  frames: {F_cells.shape[1]}")

    # ── Compute dF/F ──────────────────────────────────────────────────────
    nCells, lenT = F_cells.shape
    dFF = np.zeros((nCells, lenT))
    for c in range(nCells):
        norm_F = F_cells[c] - neu_correction * Fneu_cells[c] + neu_correction * np.nanmean(Fneu_cells[c])
        F0     = scipy.stats.mode(norm_F, nan_policy='omit').mode
        dFF[c] = (norm_F - F0) / F0 * 100

    # ── Pairwise correlations + distances ─────────────────────────────────
    centroids = np.array([s['med'] for s in stat_cells])
    corr_mat  = np.corrcoef(dFF)
    perm_mat  = np.array(list(itertools.combinations(range(nCells), 2)))
    cc_vec    = corr_mat[perm_mat[:, 0], perm_mat[:, 1]]
    diff      = centroids[perm_mat[:, 0]] - centroids[perm_mat[:, 1]]
    dist_vec  = np.sqrt((diff ** 2).sum(axis=1))

    # ── Build edges and connected components ──────────────────────────────
    dist_mask = (dist_vec < max_distance) if use_max_distance else np.ones(len(cc_vec), dtype=bool)
    edge_mask = (cc_vec > cc_thresh) & dist_mask

    adjacency = defaultdict(set)
    for idx in np.where(edge_mask)[0]:
        a, b = perm_mat[idx]
        adjacency[a].add(b)
        adjacency[b].add(a)

    n_corr_only = int(np.sum(cc_vec > cc_thresh))
    n_edges     = int(np.sum(edge_mask))
    if use_max_distance:
        print(f"Pairs above cc>{cc_thresh}: {n_corr_only}  |  also within {max_distance}px: {n_edges}  |  removed by distance: {n_corr_only - n_edges}")
    else:
        print(f"Pairs above cc>{cc_thresh}: {n_edges}  (no distance constraint)")

    visited, kept_groups = set(), []
    for node in range(nCells):
        if node not in visited:
            stack, group = [node], set()
            while stack:
                n = stack.pop()
                if n not in visited:
                    visited.add(n)
                    group.add(n)
                    stack.extend(adjacency[n] - visited)
            kept_groups.append(sorted(list(group)))

    # ── Average traces within groups ──────────────────────────────────────
    dFF_merged = np.array([np.mean(dFF[g, :], axis=0) for g in kept_groups])

    # ── Denoise + infer spikes ────────────────────────────────────────────
    denoised_dFF, spikes = TwoP.calc_inf_spikes(dFF_merged, fps=twop_rate)

    n_merged       = dFF_merged.shape[0]
    n_multi_groups = sum(1 for g in kept_groups if len(g) > 1)
    print(f"Original: {nCells}  →  merged: {n_merged}  |  multi-ROI groups: {n_multi_groups}  |  largest: {max(len(g) for g in kept_groups)}")

    # ── Figures ───────────────────────────────────────────────────────────
    fig_save_path = os.path.join(suite2p_path, 'merging_figures')
    os.makedirs(fig_save_path, exist_ok=True)

    mean_img = ops.get('meanImg', np.zeros((ops['Ly'], ops['Lx'])))
    vmin_img = np.percentile(mean_img, 1)
    vmax_img = np.percentile(mean_img, 99)

    # Before / after spatial map
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    for ax in axes:
        ax.imshow(mean_img, cmap='gray', vmin=vmin_img, vmax=vmax_img)
        ax.axis('off')

    for i in range(len(stat_cells)):
        axes[0].plot(stat_cells[i]['xpix'], stat_cells[i]['ypix'], 'c.', ms=0.3, alpha=0.5)
    axes[0].set_title(f'Before merging: {nCells} ROIs')

    cmap_g = plt.cm.tab10
    color_idx = 0
    for gi, g in enumerate(kept_groups):
        if len(g) == 1:
            axes[1].plot(stat_cells[g[0]]['xpix'], stat_cells[g[0]]['ypix'], '.', color='gray', ms=0.3, alpha=0.4)
        else:
            color = cmap_g(color_idx % 10)
            color_idx += 1
            for gx in g:
                axes[1].plot(stat_cells[gx]['xpix'], stat_cells[gx]['ypix'], '.', color=color, ms=0.5, alpha=0.8)
    axes[1].set_title(f'After merging: {n_merged} axons')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_path, 'roi_before_after_merging.png'), bbox_inches='tight', dpi=150)
    plt.close()

    # Merged group traces (4×4 grid per figure)
    multi_groups = [(gi, g) for gi, g in enumerate(kept_groups) if len(g) > 1]
    if multi_groups:
        n_rows, n_cols    = 4, 4
        groups_per_fig    = n_rows * n_cols
        n_figs            = int(np.ceil(len(multi_groups) / groups_per_fig))
        n_frames_show     = min(int(100 * twop_rate), lenT)
        time              = np.arange(n_frames_show) / twop_rate

        for fig_i in range(n_figs):
            start = fig_i * groups_per_fig
            end   = min(start + groups_per_fig, len(multi_groups))
            fig, axes_arr = plt.subplots(n_rows, n_cols, figsize=(16, 12), dpi=100)
            for j, ax in enumerate(axes_arr.flatten()):
                idx = start + j
                if idx < end:
                    gi, g = multi_groups[idx]
                    for gx in g:
                        ax.plot(time, dFF[gx, :n_frames_show], alpha=0.4, lw=0.5)
                    ax.plot(time, dFF_merged[gi, :n_frames_show], 'k-', lw=1)
                    ax.set_title(f'Group {gi}: {len(g)} ROIs {g}', fontsize=7)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.tick_params(labelsize=6)
                else:
                    ax.axis('off')
            fig.suptitle(f'Merged groups {start+1}–{end} of {len(multi_groups)}', fontsize=11)
            fig.supxlabel('Time (s)', fontsize=9)
            fig.supylabel('dF/F (%)', fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_save_path, f'merged_groups_fig{fig_i+1:02d}.png'), bbox_inches='tight', dpi=100)
            plt.close()

    print(f"Figures saved → {fig_save_path}")

    # ── Update iscell.npy ─────────────────────────────────────────────────
    iscell_premerge_path = os.path.join(suite2p_path, 'iscell_premerge.npy')
    iscell_path          = os.path.join(suite2p_path, 'iscell.npy')

    if not os.path.exists(iscell_premerge_path):
        np.save(iscell_premerge_path, iscell)
        print(f"Backup saved → {iscell_premerge_path}")
    else:
        print(f"Backup already exists, skipping: {iscell_premerge_path}")

    iscell_updated = iscell.copy()
    n_removed = 0
    for g in kept_groups:
        if len(g) > 1:
            kurt_vals = [scipy.stats.kurtosis(dFF[idx, :], fisher=True) for idx in g]
            best_idx  = g[np.argmax(kurt_vals)]
            for idx in g:
                if idx != best_idx:
                    iscell_updated[original_indices[idx], 0] = 0
                    n_removed += 1

    np.save(iscell_path, iscell_updated)
    n_before = int(np.sum(iscell[:, 0] == 1))
    n_after  = int(np.sum(iscell_updated[:, 0] == 1))
    print(f"iscell updated: {n_before} → {n_after} (removed {n_removed} redundant ROIs)")
    print(f"Saved → {iscell_path}")

    return {
        'suite2p_path': suite2p_path,
        'n_original': nCells,
        'n_merged': n_merged,
        'n_multi_groups': n_multi_groups,
        'n_removed': n_removed,
    }


# ============================================================================
# BATCH LOOP
# ============================================================================

if __name__ == "__main__":

    successful, failed = [], []

    for suite2p_path in SUITE2P_PATHS:
        # Skip check
        if SKIP_IF_DONE and os.path.exists(os.path.join(suite2p_path, 'iscell_premerge.npy')):
            print(f"[SKIP] Already merged: {suite2p_path}")
            continue

        try:
            result = run_axon_merge(suite2p_path)
            successful.append(result)
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {suite2p_path}\n{traceback.format_exc()}")
            failed.append((suite2p_path, str(e)))

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"BATCH COMPLETE  —  {len(successful)} succeeded  /  {len(failed)} failed")
    print(f"{'='*80}")
    if successful:
        print(f"\n{'Session':<55} {'orig':>6} {'merged':>7} {'groups':>7} {'removed':>8}")
        print("-" * 80)
        for r in successful:
            label = os.path.basename(os.path.dirname(os.path.dirname(r['suite2p_path'])))
            print(f"{label:<55} {r['n_original']:>6} {r['n_merged']:>7} {r['n_multi_groups']:>7} {r['n_removed']:>8}")
    if failed:
        print("\nFailed:")
        for path, err in failed:
            print(f"  {path}\n    {err}")
