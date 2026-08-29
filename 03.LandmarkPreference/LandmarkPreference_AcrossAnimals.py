"""
LandmarkPreference_AcrossAnimals.py
Aggregate landmark preference results across all animals.

Figures saved to D:/V1_SpatialModulation/2p/V1_prism/landmark_analysis/:
  A. population_trajectory    — mean±SEM proportion per landmark per layer
  B. proportion_heatmap_grid  — Grand Average + Day1-7 heatmaps (layers × landmarks,
                                 color = proportion, annotated with mean±SEM, n=cells)
  C. day1_distribution        — mean±SEM + dots per landmark on Day 1
  D. early_vs_late_proportion — proportion per landmark early vs late, per layer
  E. layer_comparison         — proportion per landmark per layer at Day 1 vs last day
  F. per_animal_proportion    — individual animal L2/3 proportion per landmark + mean
  G. sup_vs_deep_proportion   — proportion per landmark superficial vs deep across days
  H. stats_table              — Kruskal-Wallis + Mann-Whitney + Cliff's delta on proportions
  I. summary                  — proportion trajectory + bias/entropy secondary

JSY, 2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
from scipy.stats import kruskal, mannwhitneyu
from glob import glob
import h5py


# ─────────────────────────── constants ───────────────────────────────────────

LAYER_ORDER  = ['L2/3', 'L4', 'L5', 'L6']
LAYER_COLORS = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
LANDMARK_COLORS    = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']
LANDMARK_POSITIONS = [25, 55, 85, 115]
N_LM = len(LANDMARK_POSITIONS)

ANIMAL_DIRS = {
    'JSY040': r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging',
    # 'JSY041': r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging',
    # 'JSY044': r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging',
    'JSY051': r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging',
    'JSY052': r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging',
    'JSY054': r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging',
    'JSY055': r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging',
}

OUTPUT_DIR = r'D:\V1_SpatialModulation\2p\V1_prism\landmark_analysis_WtO4144'


# ─────────────────────────── helpers ─────────────────────────────────────────

def bias_index(proportions):
    p = np.asarray(proportions, dtype=float)
    if np.all(p == 0) or np.any(np.isnan(p)):
        return np.nan
    return float(np.max(p) - 1.0 / len(p))


def entropy_norm(proportions):
    p = np.asarray(proportions, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return np.nan
    H = -np.sum(p * np.log(p))
    return float(H / np.log(N_LM))


def cliffs_delta(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    greater = sum(xi > yj for xi in x for yj in y)
    less    = sum(xi < yj for xi in x for yj in y)
    return (greater - less) / (n1 * n2)


def _stars(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def _mean_sem(vals):
    vals = np.asarray([v for v in vals if not np.isnan(v)], dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan
    return np.mean(vals), np.std(vals) / np.sqrt(len(vals))


# ─────────────────────────── data loading ────────────────────────────────────

def load_all_animals(animal_dirs=None, h5_pattern='**/*_landmark_preferences.h5'):
    """
    Load landmark preference h5 files for all animals.

    Returns
    -------
    all_data : dict
        {animal_id: {day: {layer: {proportions, counts, n_cells, bias_index, entropy}}}}
    """
    if animal_dirs is None:
        animal_dirs = ANIMAL_DIRS

    all_data = {}

    for animal_id, adir in animal_dirs.items():
        if not os.path.isdir(adir):
            print(f'  Skipping {animal_id} — directory not found')
            continue

        h5_files = glob(os.path.join(adir, h5_pattern), recursive=True)
        if not h5_files:
            print(f'  Skipping {animal_id} — no h5 files found')
            continue

        print(f'  {animal_id}: {len(h5_files)} sessions')
        animal_data = {}

        for h5_path in sorted(h5_files):
            m_day = re.search(r'Day(\d+)', h5_path, re.IGNORECASE)
            if m_day is None:
                continue
            day = int(m_day.group(1))

            try:
                day_data = {}
                with h5py.File(h5_path, 'r') as f:
                    if 'full_session' not in f:
                        continue
                    for safe_key in f['full_session'].keys():
                        grp  = f['full_session'][safe_key]
                        orig = str(grp.attrs.get('original_name', safe_key.replace('_', '/')))
                        props  = grp['landmark_proportions'][:] if 'landmark_proportions' in grp else None
                        counts = grp['landmark_counts'][:].astype(int) if 'landmark_counts' in grp else None
                        n_cells = int(grp.attrs.get('n_cells', 0))
                        if props is not None:
                            day_data[orig] = {
                                'proportions': props,
                                'counts':      counts,
                                'n_cells':     n_cells,
                                'bias_index':  bias_index(props),
                                'entropy':     entropy_norm(props),
                            }
                animal_data[day] = day_data
            except Exception as e:
                print(f'    ERROR {h5_path}: {e}')

        if animal_data:
            all_data[animal_id] = animal_data

    print(f'\n  Loaded {len(all_data)} animals.')
    return all_data


# ─────────────────────────── aggregation helpers ─────────────────────────────

def _all_days(all_data):
    days = set()
    for anim_data in all_data.values():
        days.update(anim_data.keys())
    return sorted(days)


def _aggregate_for_days(all_data, target_days=None):
    """
    For given days (None = all days), return per layer×landmark:
      - list of per-animal proportions
      - total cell count (sum across animals/sessions)
      - number of animals contributing

    Returns: {layer: {'props': [N_LM lists], 'counts': [N_LM ints], 'n_animals': int}}
    """
    result = {layer: {'props': [[] for _ in range(N_LM)],
                      'counts': np.zeros(N_LM, dtype=int),
                      'n_animals': set()} for layer in LAYER_ORDER}

    for animal_id, anim_data in all_data.items():
        for day, day_data in anim_data.items():
            if target_days is not None and day not in target_days:
                continue
            for layer in LAYER_ORDER:
                ld = day_data.get(layer)
                if ld is None or ld['proportions'] is None:
                    continue
                for lm_i in range(N_LM):
                    result[layer]['props'][lm_i].append(float(ld['proportions'][lm_i]))
                    if ld['counts'] is not None:
                        result[layer]['counts'][lm_i] += int(ld['counts'][lm_i])
                result[layer]['n_animals'].add(animal_id)

    # Convert n_animals sets to counts
    for layer in LAYER_ORDER:
        result[layer]['n_animals'] = len(result[layer]['n_animals'])

    return result


def _layer_props_by_day(all_data, layer):
    """Returns {day: [proportions_array per animal]}."""
    result = {}
    for anim_data in all_data.values():
        for day, day_data in anim_data.items():
            ld = day_data.get(layer)
            if ld is not None and ld['proportions'] is not None:
                result.setdefault(day, []).append(ld['proportions'])
    return result


def _layer_bias_by_day(all_data, layer):
    result = {}
    for anim_data in all_data.values():
        for day, day_data in anim_data.items():
            ld = day_data.get(layer)
            if ld is not None and not np.isnan(ld['bias_index']):
                result.setdefault(day, []).append(ld['bias_index'])
    return result


def _layer_entropy_by_day(all_data, layer):
    result = {}
    for anim_data in all_data.values():
        for day, day_data in anim_data.items():
            ld = day_data.get(layer)
            if ld is not None and not np.isnan(ld['entropy']):
                result.setdefault(day, []).append(ld['entropy'])
    return result


# ─────────────────────────── heatmap panel helper ────────────────────────────

def _draw_proportion_heatmap(ax, agg, title, cmap, vmin=0, vmax=0.8):
    """
    Draw a single layers × landmarks heatmap panel from aggregated data.
    Each cell shows: mean ± SEM on top, (n=cells) below.
    """
    layers = LAYER_ORDER
    n_layers = len(layers)
    mat_mean = np.full((n_layers, N_LM), np.nan)
    mat_sem  = np.full((n_layers, N_LM), np.nan)
    mat_n    = np.zeros((n_layers, N_LM), dtype=int)

    for ri, layer in enumerate(layers):
        if layer not in agg:
            continue
        for lm_i in range(N_LM):
            vals = agg[layer]['props'][lm_i]
            if vals:
                mat_mean[ri, lm_i] = np.mean(vals)
                mat_sem[ri, lm_i]  = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            mat_n[ri, lm_i] = agg[layer]['counts'][lm_i]

    im = ax.imshow(mat_mean, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    for ri in range(n_layers):
        for ci in range(N_LM):
            m = mat_mean[ri, ci]
            s = mat_sem[ri, ci]
            n = mat_n[ri, ci]
            if np.isnan(m):
                ax.text(ci, ri, 'N/A', ha='center', va='center', fontsize=23, color='gray')
                continue
            text_color = 'white' if m > 0.55 else 'black'
            label = f'{m:.2f}\n±{s:.2f}\n(n={n})'
            ax.text(ci, ri, label, ha='center', va='center',
                    fontsize=23, color=text_color, linespacing=1.5)

    ax.set_xticks(range(N_LM))
    ax.set_xticklabels([f'LD{i+1}\n({LANDMARK_POSITIONS[i]}cm)' for i in range(N_LM)], fontsize=23)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layers, fontsize=23)
    for i, layer in enumerate(layers):
        ax.get_yticklabels()[i].set_color(LAYER_COLORS.get(layer, 'k'))
    # ax.set_xlabel('Landmark', fontsize=23)
    # ax.set_ylabel('Layer', fontsize=23)
    ax.set_title(title, fontsize=25, fontweight='bold')

    return im


# ─────────────────────────── Figure A: population trajectory ─────────────────

def plot_population_trajectory(all_data, save_path):
    """Mean±SEM proportion per landmark per layer across days."""
    days    = _all_days(all_data)
    layers  = LAYER_ORDER
    n_layers = len(layers)

    fig, axes = plt.subplots(1, n_layers, figsize=(5.5 * n_layers, 6), sharey=True)
    if n_layers == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        props_by_day = _layer_props_by_day(all_data, layer)
        for lm_i in range(N_LM):
            means, sems, xs = [], [], []
            for day in days:
                vals = [p[lm_i] for p in props_by_day.get(day, []) if not np.isnan(p[lm_i])]
                if vals:
                    means.append(np.mean(vals))
                    sems.append(np.std(vals) / np.sqrt(len(vals)))
                    xs.append(day)
            if xs:
                ax.errorbar(xs, means, yerr=sems,
                            color=LANDMARK_COLORS[lm_i], marker='o',
                            linewidth=2, capsize=3,
                            label=f'LD{lm_i+1} ({LANDMARK_POSITIONS[lm_i]}cm)')
        ax.axhline(1 / N_LM, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_title(layer, color=LAYER_COLORS.get(layer, 'k'), fontsize=16, fontweight='bold')
        ax.set_xlabel('Day', fontsize=14)
        ax.set_ylim(0, 1)
        ax.set_xticks(days)
        ax.tick_params(labelsize=13)

    axes[0].set_ylabel('Mean proportion of cells (± SEM)', fontsize=14)
    axes[-1].legend(fontsize=13, loc='upper right')
    fig.suptitle('Population landmark preference trajectory', fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'A_population_trajectory.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: A_population_trajectory')


# ─────────────────────────── Figure B: proportion heatmap grid ───────────────

def plot_proportion_heatmap_grid(all_data, save_path):
    """
    Grand Average + one panel per day (2 rows × 4 cols).
    Each panel: layers × landmarks heatmap, color = mean proportion.
    Cells annotated with mean ± SEM and total n.
    """
    days = _all_days(all_data)
    n_panels = 1 + len(days)          # Grand Average + per-day
    n_cols   = 4
    n_rows   = int(np.ceil(n_panels / n_cols))

    cmap = plt.cm.YlOrRd

    fig = plt.figure(figsize=(16 * n_cols, 13 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols + 1,
                             width_ratios=[1] * n_cols + [0.04],
                             hspace=0.25, wspace=0.15)

    # --- Grand Average ---
    agg_all = _aggregate_for_days(all_data, target_days=None)
    ax0  = fig.add_subplot(gs[0, 0])
    n_animals_total = len(all_data)
    n_sessions_total = sum(len(anim) for anim in all_data.values())
    title0 = 'Grand Average\n(All Animals, All Sessions)'
    _draw_proportion_heatmap(ax0, agg_all, title0, cmap)

    # --- Per-day panels ---
    for pi, day in enumerate(days):
        row = (pi + 1) // n_cols
        col = (pi + 1) % n_cols
        ax  = fig.add_subplot(gs[row, col])
        n_animals_day = sum(1 for anim in all_data.values() if day in anim)
        agg_day = _aggregate_for_days(all_data, target_days=[day])
        title = f'Day{day}\n({n_animals_day} animals)'
        _draw_proportion_heatmap(ax, agg_day, title, cmap)

    # Hide unused panels
    total_slots = n_rows * n_cols
    for empty in range(n_panels, total_slots):
        row = empty // n_cols
        col = empty % n_cols
        fig.add_subplot(gs[row, col]).set_visible(False)

    # Shared colorbar
    cbar_ax = fig.add_subplot(gs[:, -1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=0.8))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Proportion of Cells', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    fig.suptitle(f'Landmark Preference — Proportion of Cells per Layer  (n={n_animals_total} animals, {n_sessions_total} sessions)',
                 fontsize=18, fontweight='bold', y=1.01)
    fig.savefig(os.path.join(save_path, 'B_proportion_heatmap_grid.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: B_proportion_heatmap_grid')


# ─────────────────────────── Figure C: Day-1 distribution ───────────────────

def plot_day1_distribution(all_data, save_path):
    """Mean±SEM + individual animal dots per landmark on Day 1, per layer."""
    day1_options = [d for d in _all_days(all_data) if d <= 2]
    if not day1_options:
        print('  Skipped: C_day1_distribution (no Day 1 data)')
        return
    d1 = day1_options[0]

    fig, axes = plt.subplots(1, len(LAYER_ORDER), figsize=(5.5 * len(LAYER_ORDER), 6), sharey=True)

    for ax, layer in zip(axes, LAYER_ORDER):
        props_by_animal = []
        for anim_data in all_data.values():
            ld = anim_data.get(d1, {}).get(layer)
            if ld is not None and ld['proportions'] is not None:
                props_by_animal.append(ld['proportions'])

        if not props_by_animal:
            ax.set_visible(False)
            continue

        mat   = np.array(props_by_animal)
        means = mat.mean(axis=0)
        sems  = mat.std(axis=0) / np.sqrt(len(mat))
        x     = np.arange(N_LM)

        ax.bar(x, means, yerr=sems, color=LANDMARK_COLORS, capsize=4, alpha=0.85)
        rng = np.random.default_rng(0)
        for ai in range(mat.shape[0]):
            ax.plot(x + rng.uniform(-0.15, 0.15, N_LM),
                    mat[ai], 'k.', markersize=5, alpha=0.5)
        ax.axhline(1 / N_LM, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'LD{i+1}\n({LANDMARK_POSITIONS[i]}cm)' for i in range(N_LM)], fontsize=14)
        ax.set_ylim(0, 1)
        ax.set_title(layer, color=LAYER_COLORS.get(layer, 'k'), fontsize=16, fontweight='bold')
        ax.set_xlabel('Landmark', fontsize=14)
        ax.tick_params(labelsize=13)

    axes[0].set_ylabel('Proportion of cells', fontsize=14)
    fig.suptitle(f'Day {d1} landmark distribution (mean ± SEM, dots = individual animals)', fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'C_day1_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: C_day1_distribution')


# ─────────────────────────── Figure D: early vs late proportion ──────────────

def plot_early_vs_late_proportion(all_data, save_path):
    """
    Proportion per landmark, early (D1-2) vs late (D6-7), per layer.
    One subplot per layer, grouped bars per landmark with Mann-Whitney stars.
    """
    days       = _all_days(all_data)
    early_days = [d for d in days if d <= 2]
    late_days  = [d for d in days if d >= 6]

    if not early_days or not late_days:
        print('  Skipped: D_early_vs_late_proportion (insufficient days)')
        return

    agg_early = _aggregate_for_days(all_data, target_days=early_days)
    agg_late  = _aggregate_for_days(all_data, target_days=late_days)

    fig, axes = plt.subplots(1, len(LAYER_ORDER),
                              figsize=(5.5 * len(LAYER_ORDER), 6), sharey=True)
    x     = np.arange(N_LM)
    width = 0.35

    for ax, layer in zip(axes, LAYER_ORDER):
        early_means, early_sems, pvals = [], [], []
        late_means,  late_sems         = [], []

        for lm_i in range(N_LM):
            e_vals = agg_early[layer]['props'][lm_i]
            l_vals = agg_late[layer]['props'][lm_i]
            me, se = _mean_sem(e_vals)
            ml, sl = _mean_sem(l_vals)
            early_means.append(me)
            early_sems.append(se)
            late_means.append(ml)
            late_sems.append(sl)

            # if len(e_vals) >= 3 and len(l_vals) >= 3:
            #     try:
            #         _, p = mannwhitneyu(e_vals, l_vals, alternative='two-sided')
            #     except Exception:
            #         p = np.nan
            # else:
            #     p = np.nan
            # pvals.append(p)

        ax.bar(x - width / 2, early_means, width, yerr=early_sems,
               color=LANDMARK_COLORS, alpha=0.45, capsize=4,
               label=f'Early (D{min(early_days)}–{max(early_days)})')
        ax.bar(x + width / 2, late_means, width, yerr=late_sems,
               color=LANDMARK_COLORS, alpha=1.0, capsize=4,
               label=f'Late (D{min(late_days)}–{max(late_days)})')

        y_top = max([(m or 0) + (s or 0) for m, s in
                     zip(early_means + late_means, early_sems + late_sems)],
                    default=0) + 0.05
        for xi, p in enumerate(pvals):
            if not np.isnan(p):
                ax.text(xi, y_top, _stars(p), ha='center', fontsize=14)

        ax.set_xticks(x)
        ax.set_xticklabels([f'LD{i+1}\n({LANDMARK_POSITIONS[i]}cm)' for i in range(N_LM)], fontsize=14)
        ax.set_ylim(0, min(1.05, y_top + 0.1))
        ax.axhline(1 / N_LM, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(layer, color=LAYER_COLORS.get(layer, 'k'), fontsize=16, fontweight='bold')
        ax.set_xlabel('Landmark', fontsize=14)
        ax.tick_params(labelsize=13)

    axes[0].set_ylabel('Proportion of cells (± SEM)', fontsize=14)
    axes[-1].legend(fontsize=13, loc='upper right')
    fig.suptitle('Early vs Late landmark preference — proportion of cells', fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'D_early_vs_late_proportion.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: D_early_vs_late_proportion')


# ─────────────────────────── Figure E: layer comparison D1 vs Dlast ──────────

def plot_layer_comparison(all_data, save_path):
    """
    Proportion per landmark at Day 1 vs last day, all layers side-by-side.
    One subplot per landmark.
    """
    days   = _all_days(all_data)
    d1     = days[0]
    d_last = days[-1]

    agg_d1    = _aggregate_for_days(all_data, target_days=[d1])
    agg_dlast = _aggregate_for_days(all_data, target_days=[d_last])

    fig, axes = plt.subplots(1, N_LM, figsize=(5.5 * N_LM, 6), sharey=True)
    x     = np.arange(len(LAYER_ORDER))
    width = 0.35

    for lm_i, ax in enumerate(axes):
        d1_means, d1_sems     = [], []
        dlast_means, dlast_sems = [], []

        for layer in LAYER_ORDER:
            m1, s1 = _mean_sem(agg_d1[layer]['props'][lm_i])
            ml, sl = _mean_sem(agg_dlast[layer]['props'][lm_i])
            d1_means.append(m1)
            d1_sems.append(s1)
            dlast_means.append(ml)
            dlast_sems.append(sl)

        layer_colors = [LAYER_COLORS.get(layer, 'k') for layer in LAYER_ORDER]
        ax.bar(x - width / 2, d1_means, width, yerr=d1_sems,
               color=layer_colors, alpha=0.45, capsize=4, label=f'Day {d1}')
        ax.bar(x + width / 2, dlast_means, width, yerr=dlast_sems,
               color=layer_colors, alpha=1.0, capsize=4, label=f'Day {d_last}')

        ax.set_xticks(x)
        ax.set_xticklabels(LAYER_ORDER, fontsize=14)
        ax.set_ylim(0, 1)
        ax.axhline(1 / N_LM, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(f'LD{lm_i+1} ({LANDMARK_POSITIONS[lm_i]} cm)',
                     color=LANDMARK_COLORS[lm_i], fontsize=16, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=14)
        ax.tick_params(labelsize=13)

    axes[0].set_ylabel('Proportion of cells (± SEM)', fontsize=14)
    axes[-1].legend(fontsize=13)
    fig.suptitle(f'Layer comparison: proportion per landmark — Day {d1} vs Day {d_last}', fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'E_layer_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: E_layer_comparison')


# ─────────────────────────── Figure F: per-animal proportion in L2/3 ─────────

def plot_per_animal_proportion(all_data, save_path):
    """
    Individual animal proportion per landmark in L2/3 across days + bold mean.
    One subplot per landmark.
    """
    days  = _all_days(all_data)
    layer = 'L2/3'

    fig, axes = plt.subplots(1, N_LM, figsize=(5.5 * N_LM, 6), sharey=True)

    for lm_i, ax in enumerate(axes):
        # Individual animal traces
        for _, anim_data in all_data.items():
            xs, ys = [], []
            for day in days:
                ld = anim_data.get(day, {}).get(layer)
                if ld is not None and ld['proportions'] is not None:
                    xs.append(day)
                    ys.append(float(ld['proportions'][lm_i]))
            if xs:
                ax.plot(xs, ys, color=LANDMARK_COLORS[lm_i], linewidth=1,
                        alpha=0.35, marker='o', markersize=3)

        # Population mean ± SEM
        props_by_day = _layer_props_by_day(all_data, layer)
        mean_xs, mean_ys, sems = [], [], []
        for day in days:
            vals = [p[lm_i] for p in props_by_day.get(day, []) if not np.isnan(p[lm_i])]
            if vals:
                mean_xs.append(day)
                mean_ys.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
        if mean_xs:
            ax.errorbar(mean_xs, mean_ys, yerr=sems,
                        color=LANDMARK_COLORS[lm_i], linewidth=3, capsize=4,
                        marker='o', markersize=7,
                        label=f'Mean (n={len(all_data)} animals)', zorder=10)

        ax.axhline(1 / N_LM, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(f'LD{lm_i+1} ({LANDMARK_POSITIONS[lm_i]} cm)',
                     color=LANDMARK_COLORS[lm_i], fontsize=16, fontweight='bold')
        ax.set_xlabel('Day', fontsize=14)
        ax.set_ylim(0, 1)
        ax.set_xticks(days)
        ax.tick_params(labelsize=13)

    axes[0].set_ylabel('Proportion of cells (L2/3)', fontsize=14)
    axes[-1].legend(fontsize=13)
    fig.suptitle('L2/3 — individual animal proportion per landmark (thin) + mean±SEM (bold)', fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'F_per_animal_proportion.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: F_per_animal_proportion')


# ─────────────────────────── Figure G: sup vs deep proportion ────────────────

def plot_sup_vs_deep_proportion(all_data, save_path):
    """
    Proportion per landmark, superficial (L2/3+L4) vs deep (L5+L6) across days.
    One subplot per landmark.
    """
    days   = _all_days(all_data)
    groups = {
        'Superficial\n(L2/3+L4)': (['L2/3', 'L4'], '#1E88E5'),
        'Deep\n(L5+L6)':          (['L5', 'L6'],   '#E53935'),
    }

    fig, axes = plt.subplots(1, N_LM, figsize=(5.5 * N_LM, 6), sharey=True)

    for lm_i, ax in enumerate(axes):
        for gname, (glayers, gcol) in groups.items():
            means, sems, xs = [], [], []
            for day in days:
                vals = []
                for anim_data in all_data.values():
                    dd = anim_data.get(day, {})
                    for layer in glayers:
                        ld = dd.get(layer)
                        if ld is not None and ld['proportions'] is not None:
                            vals.append(float(ld['proportions'][lm_i]))
                if vals:
                    means.append(np.mean(vals))
                    sems.append(np.std(vals) / np.sqrt(len(vals)))
                    xs.append(day)
            if xs:
                ax.errorbar(xs, means, yerr=sems,
                            color=gcol, marker='o', linewidth=2.5,
                            capsize=4, label=gname)

        ax.axhline(1 / N_LM, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(f'LD{lm_i+1} ({LANDMARK_POSITIONS[lm_i]} cm)',
                     color=LANDMARK_COLORS[lm_i], fontsize=16, fontweight='bold')
        ax.set_xlabel('Day', fontsize=14)
        ax.set_ylim(0, 1)
        ax.set_xticks(days)
        ax.tick_params(labelsize=13)

    axes[0].set_ylabel('Proportion of cells (± SEM)', fontsize=14)
    axes[-1].legend(fontsize=13, loc='upper right')
    fig.suptitle('Superficial vs Deep — proportion preferring each landmark', fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'G_sup_vs_deep_proportion.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: G_sup_vs_deep_proportion')


# ─────────────────────────── Figure H: stats table ──────────────────────────

def plot_stats_table(all_data, save_path):
    """
    For each layer × landmark: Kruskal-Wallis across days + Mann-Whitney D1 vs D_last
    + Cliff's delta on the proportion of cells.
    """
    days   = _all_days(all_data)
    d1     = days[0] if days else None
    d_last = days[-1] if days else None

    rows = []
    for layer in LAYER_ORDER:
        props_by_day = _layer_props_by_day(all_data, layer)

        for lm_i in range(N_LM):
            # Proportion values per day
            per_day = {day: [p[lm_i] for p in props_by_day.get(day, [])]
                       for day in days}

            groups = [np.array(v) for v in per_day.values() if len(v) > 0]
            if len(groups) >= 2:
                try:
                    _, kw_p = kruskal(*groups)
                except Exception:
                    kw_p = np.nan
            else:
                kw_p = np.nan

            g1 = per_day.get(d1, [])
            gl = per_day.get(d_last, [])
            if len(g1) >= 2 and len(gl) >= 2:
                try:
                    _, mw_p = mannwhitneyu(g1, gl, alternative='two-sided')
                    cd = cliffs_delta(gl, g1)
                except Exception:
                    mw_p, cd = np.nan, np.nan
            else:
                mw_p, cd = np.nan, np.nan

            m1, s1 = _mean_sem(g1)
            ml, sl = _mean_sem(gl)

            rows.append({
                'Layer':              layer,
                'Landmark':           f'LD{lm_i+1} ({LANDMARK_POSITIONS[lm_i]}cm)',
                'D1 mean±SEM':        f'{m1:.2f}±{s1:.2f}' if not np.isnan(m1) else 'n/a',
                f'D{d_last} mean±SEM': f'{ml:.2f}±{sl:.2f}' if not np.isnan(ml) else 'n/a',
                'KW p':               f'{kw_p:.4f}' if not np.isnan(kw_p) else 'n/a',
                'KW sig':             _stars(kw_p) if not np.isnan(kw_p) else 'n/a',
                'MW p (D1 vs Dlast)': f'{mw_p:.4f}' if not np.isnan(mw_p) else 'n/a',
                'MW sig':             _stars(mw_p) if not np.isnan(mw_p) else 'n/a',
                "Cliff's delta":      f'{cd:.3f}' if not np.isnan(cd) else 'n/a',
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_path, 'H_stats_table.csv'), index=False)
    print('  Saved: H_stats_table.csv')

    n_rows = len(df)
    fig, ax = plt.subplots(figsize=(20, 2.8 + 0.5 * n_rows))
    ax.axis('off')
    col_labels = list(df.columns)
    table = ax.table(cellText=df.values, colLabels=col_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.0)
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    ax.set_title(f'Proportion statistics: D{d1} vs D{d_last} (per layer × landmark)',
                 fontsize=16, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'H_stats_table.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: H_stats_table.png')
    print(df.to_string(index=False))
    return df


# ─────────────────────────── Figure I: summary panel ─────────────────────────

def plot_summary(all_data, save_path):
    """Combined summary: proportion trajectory (top) + bias/entropy secondary (bottom)."""
    days   = _all_days(all_data)
    layers = LAYER_ORDER

    fig = plt.figure(figsize=(max(12, 2.4 * len(days)), 16))
    gs  = gridspec.GridSpec(3, 1, hspace=0.50)

    # --- Top: proportion trajectory for all landmarks, L2/3 ---
    ax_prop = fig.add_subplot(gs[0])
    props_by_day = _layer_props_by_day(all_data, 'L2/3')
    for lm_i in range(N_LM):
        xs, means, sems = [], [], []
        for day in days:
            vals = [p[lm_i] for p in props_by_day.get(day, []) if not np.isnan(p[lm_i])]
            if vals:
                xs.append(day)
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
        if xs:
            ax_prop.errorbar(xs, means, yerr=sems,
                             color=LANDMARK_COLORS[lm_i], marker='o', linewidth=2,
                             capsize=3, label=f'LD{lm_i+1} ({LANDMARK_POSITIONS[lm_i]}cm)')
    ax_prop.axhline(1 / N_LM, color='gray', linestyle='--', linewidth=0.8)
    ax_prop.set_ylabel('Proportion of cells (± SEM)', fontsize=14)
    ax_prop.set_ylim(0, 1)
    ax_prop.set_xticks(days)
    ax_prop.tick_params(labelsize=13)
    ax_prop.legend(fontsize=13, title='Landmark')
    ax_prop.set_title('L2/3 — Proportion per landmark across days', fontsize=16)

    # --- Mid: bias index per layer ---
    ax_bias = fig.add_subplot(gs[1])
    for layer in layers:
        bbd = _layer_bias_by_day(all_data, layer)
        xs, means, sems = [], [], []
        for day in days:
            vals = bbd.get(day, [])
            if vals:
                xs.append(day)
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
        if xs:
            ax_bias.errorbar(xs, means, yerr=sems,
                             color=LAYER_COLORS.get(layer, 'k'), marker='o',
                             linewidth=2, capsize=3, label=layer)
    ax_bias.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_bias.set_ylabel('Bias Index (± SEM)', fontsize=14)
    ax_bias.set_ylim(-0.05, 0.75)
    ax_bias.set_xticks(days)
    ax_bias.tick_params(labelsize=13)
    ax_bias.legend(fontsize=13)
    ax_bias.set_title('Landmark Bias Index per layer', fontsize=16)

    # --- Bottom: entropy per layer ---
    ax_ent = fig.add_subplot(gs[2])
    for layer in layers:
        ebd = _layer_entropy_by_day(all_data, layer)
        xs, means, sems = [], [], []
        for day in days:
            vals = ebd.get(day, [])
            if vals:
                xs.append(day)
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
        if xs:
            ax_ent.errorbar(xs, means, yerr=sems,
                            color=LAYER_COLORS.get(layer, 'k'), marker='o',
                            linewidth=2, capsize=3, label=layer)
    ax_ent.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, label='Uniform')
    ax_ent.set_ylabel('Norm. Entropy (± SEM)', fontsize=14)
    ax_ent.set_ylim(0, 1.15)
    ax_ent.set_xlabel('Day', fontsize=14)
    ax_ent.set_xticks(days)
    ax_ent.tick_params(labelsize=13)
    ax_ent.legend(fontsize=13)
    ax_ent.set_title('Normalized Shannon Entropy per layer', fontsize=16)

    n_animals = len(all_data)
    fig.suptitle(f'Landmark Preference — Population Summary (n={n_animals} animals)',
                 fontsize=18, fontweight='bold')
    fig.savefig(os.path.join(save_path, 'I_summary.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: I_summary')


# ─────────────────────────── main runner ────────────────────────────────────

def run_across_animals_analysis(animal_dirs=None, output_dir=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f'\n{"="*70}')
    print('ACROSS-ANIMALS LANDMARK PREFERENCE ANALYSIS')
    print(f'Output: {output_dir}')
    print(f'{"="*70}\n')

    all_data = load_all_animals(animal_dirs)

    if not all_data:
        print('No data loaded — exiting.')
        return

    print(f'\n  Animals: {list(all_data.keys())}')
    print(f'  Days observed: {_all_days(all_data)}\n')

    plot_population_trajectory(all_data, output_dir)
    plot_proportion_heatmap_grid(all_data, output_dir)
    plot_day1_distribution(all_data, output_dir)
    plot_early_vs_late_proportion(all_data, output_dir)
    plot_layer_comparison(all_data, output_dir)
    plot_per_animal_proportion(all_data, output_dir)
    plot_sup_vs_deep_proportion(all_data, output_dir)
    plot_stats_table(all_data, output_dir)
    plot_summary(all_data, output_dir)

    print(f'\n  All figures saved to {output_dir}')
    return all_data


# ─────────────────────────── entry point ────────────────────────────────────

if __name__ == '__main__':
    run_across_animals_analysis()
