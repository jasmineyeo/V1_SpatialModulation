"""
LandmarkPreference_WithinAnimal.py
Analyze landmark preference development across days within a single animal.

Figures saved to {animal_dir}/LandmarkPreference/:
  A. stacked_bars      — layer x day grid of horizontal stacked proportion bars
  B. trajectory        — proportion per landmark per layer across days (4 subplots)
  C. day1_vs_last      — side-by-side stacked bars: Day 1 vs last day, per layer
  D. dominant_map      — color grid: winning landmark per layer per day
  E. bias_index        — max(p) - 1/N per layer across days
  F. entropy           — normalized Shannon entropy per layer across days
  G. early_vs_late     — first 2 vs last 2 days grouped bar chart
  H. within_session    — intra-session dynamics (preference_by_block)
  I. summary           — bias index + entropy trajectories + JSD stability

JSY, 2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.stats import kruskal, mannwhitneyu
from scipy.spatial.distance import jensenshannon
import h5py
from glob import glob


# ─────────────────────────── constants ───────────────────────────────────────

LAYER_ORDER  = ['L2/3', 'L4', 'L5', 'L6']
LAYER_COLORS = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
LANDMARK_COLORS    = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']
LANDMARK_POSITIONS = [25, 55, 85, 115]
N_LM = len(LANDMARK_POSITIONS)

ANIMAL_DIRS = {
    'JSY040': r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging',
    'JSY041': r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging',
    'JSY044': r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging',
    'JSY051': r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging',
    'JSY052': r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging',
    'JSY054': r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging',
    'JSY055': r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging',
}


# ─────────────────────────── helpers ─────────────────────────────────────────

def _safe_layer_name(name):
    """Convert 'L2/3' → 'L2_3' (HDF5 key sanitization)."""
    return str(name).replace('/', '_').replace('\\', '_')


def bias_index(proportions):
    """max(p) - 1/N_LM.  0 = uniform, 1-1/N = fully selective."""
    p = np.asarray(proportions, dtype=float)
    if np.all(p == 0):
        return np.nan
    return float(np.max(p) - 1.0 / len(p))


def entropy_norm(proportions):
    """Normalized Shannon entropy.  1 = uniform, 0 = fully selective."""
    p = np.asarray(proportions, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return np.nan
    H = -np.sum(p * np.log(p))
    H_max = np.log(N_LM)
    return float(H / H_max) if H_max > 0 else np.nan


def jsd_adjacent(prop_matrix):
    """Jensen-Shannon divergence between adjacent rows of prop_matrix [N_days, N_LM].
    Returns array of length N_days-1."""
    jsd = []
    for i in range(len(prop_matrix) - 1):
        p, q = prop_matrix[i], prop_matrix[i + 1]
        jsd.append(float(jensenshannon(p, q) ** 2))   # squared → [0,1] JSD
    return np.array(jsd)


def _dominant_index(proportions):
    """Index (0-based) of the dominant landmark."""
    p = np.asarray(proportions, dtype=float)
    if np.all(p == 0):
        return -1
    return int(np.argmax(p))


# ─────────────────────────── data loading ────────────────────────────────────

def load_animal_landmark_data(animal_dir, h5_pattern='**/*_landmark_preferences.h5'):
    """
    Load all landmark preference h5 files for one animal.

    Returns
    -------
    sessions_data : dict  {day_int: session_dict}
    animal_id     : str
    """
    h5_files = glob(os.path.join(animal_dir, h5_pattern), recursive=True)

    if not h5_files:
        print(f'  WARNING: no landmark_preferences.h5 files found in {animal_dir}')
        return {}, None

    print(f'  Found {len(h5_files)} h5 files')

    sessions_data = {}
    animal_id = None

    for h5_path in sorted(h5_files):
        m_animal = re.search(r'(JSY\d+)', h5_path)
        m_day    = re.search(r'Day(\d+)', h5_path, re.IGNORECASE)
        if m_day is None:
            continue
        day = int(m_day.group(1))
        if animal_id is None and m_animal:
            animal_id = m_animal.group(1)

        print(f'    Day {day}: {os.path.basename(h5_path)}')

        try:
            session = {
                'day': day,
                'h5_path': h5_path,
                'layers': {},
                'dynamics': None,
            }

            with h5py.File(h5_path, 'r') as f:
                session['session_id'] = str(f.attrs.get('session_id', f'Day{day}'))
                session['date']       = str(f.attrs.get('date', 'unknown'))

                # --- full-session layer data ---
                if 'full_session' in f:
                    for safe_key in f['full_session'].keys():
                        grp = f['full_session'][safe_key]
                        orig = str(grp.attrs.get('original_name', safe_key.replace('_', '/')))
                        props = grp['landmark_proportions'][:] if 'landmark_proportions' in grp else None
                        counts = grp['landmark_counts'][:] if 'landmark_counts' in grp else None
                        n_cells = int(grp.attrs.get('n_cells', 0))
                        session['layers'][orig] = {
                            'proportions': props,
                            'counts':      counts,
                            'n_cells':     n_cells,
                            'bias_index':  bias_index(props) if props is not None else np.nan,
                            'entropy':     entropy_norm(props) if props is not None else np.nan,
                        }

                # --- within-session dynamics ---
                if 'dynamics' in f:
                    dgrp = f['dynamics']
                    n_blocks = int(dgrp.attrs.get('n_blocks', 0))
                    lm_pos   = dgrp['landmark_positions'][:] if 'landmark_positions' in dgrp else np.array(LANDMARK_POSITIONS)
                    by_block = {}
                    for layer in LAYER_ORDER:
                        dset_name = f'{_safe_layer_name(layer)}_preference_by_block'
                        if dset_name in dgrp:
                            by_block[layer] = dgrp[dset_name][:]   # [N_blocks, N_LM]
                    session['dynamics'] = {
                        'n_blocks':          n_blocks,
                        'landmark_positions': lm_pos,
                        'preference_by_block': by_block,
                    }

            sessions_data[day] = session

        except Exception as e:
            print(f'    ERROR loading {h5_path}: {e}')

    return sessions_data, animal_id


# ─────────────────────────── helper: sorted days ─────────────────────────────

def _sorted_days(sessions_data):
    return sorted(sessions_data.keys())


def _prop_matrix(sessions_data, layer):
    """Return [N_days, N_LM] array of proportions (NaN if missing)."""
    days = _sorted_days(sessions_data)
    mat = np.full((len(days), N_LM), np.nan)
    for i, day in enumerate(days):
        ld = sessions_data[day]['layers'].get(layer)
        if ld is not None and ld['proportions'] is not None:
            mat[i] = ld['proportions']
    return mat


# ─────────────────────────── Figure A: stacked bars grid ─────────────────────

def plot_stacked_bars(sessions_data, animal_id, save_path):
    """Layers (rows) × days (cols) grid of horizontal stacked proportion bars."""
    days = _sorted_days(sessions_data)
    layers = [l for l in LAYER_ORDER if any(l in sessions_data[d]['layers'] for d in days)]
    n_layers = len(layers)
    n_days   = len(days)

    fig, axes = plt.subplots(n_layers, n_days,
                              figsize=(2.5 * n_days, 2.2 * n_layers),
                              sharey='row', sharex='col')
    if n_layers == 1:
        axes = axes[np.newaxis, :]
    if n_days == 1:
        axes = axes[:, np.newaxis]

    lm_labels = [f'L{p}' for p in LANDMARK_POSITIONS]

    for ri, layer in enumerate(layers):
        for ci, day in enumerate(days):
            ax = axes[ri, ci]
            ld = sessions_data[day]['layers'].get(layer)
            if ld is None or ld['proportions'] is None:
                ax.set_visible(False)
                continue
            props = ld['proportions']
            left = 0.0
            for lm_i, p in enumerate(props):
                ax.barh(0, p, left=left, color=LANDMARK_COLORS[lm_i], height=0.6)
                if p > 0.08:
                    ax.text(left + p / 2, 0, f'{p:.0%}',
                            ha='center', va='center', fontsize=7, color='white', fontweight='bold')
                left += p
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([0, 0.5, 1])
            ax.tick_params(labelsize=7)
            if ri == 0:
                ax.set_title(f'Day {day}', fontsize=9)
            if ci == 0:
                ax.set_ylabel(layer, color=LAYER_COLORS.get(layer, 'k'), fontsize=9, fontweight='bold')

    # legend
    patches = [mpatches.Patch(color=LANDMARK_COLORS[i], label=f'{LANDMARK_POSITIONS[i]} cm')
               for i in range(N_LM)]
    fig.legend(handles=patches, loc='lower center', ncol=N_LM, fontsize=8,
               title='Landmark position', bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'{animal_id} — Landmark proportion per layer × day', fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(os.path.join(save_path, f'{animal_id}_A_stacked_bars.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: A_stacked_bars')


# ─────────────────────────── Figure B: trajectory ────────────────────────────

def plot_trajectory(sessions_data, animal_id, save_path):
    """Proportion per landmark across days, one subplot per layer."""
    days = _sorted_days(sessions_data)
    layers = [l for l in LAYER_ORDER if any(l in sessions_data[d]['layers'] for d in days)]

    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 4), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        mat = _prop_matrix(sessions_data, layer)   # [N_days, N_LM]
        for lm_i in range(N_LM):
            vals = mat[:, lm_i]
            valid = ~np.isnan(vals)
            if valid.any():
                ax.plot(np.array(days)[valid], vals[valid],
                        color=LANDMARK_COLORS[lm_i], marker='o', linewidth=1.8,
                        label=f'{LANDMARK_POSITIONS[lm_i]} cm')
        ax.axhline(1 / N_LM, color='gray', linestyle='--', linewidth=0.8, alpha=0.6,
                   label='Chance')
        ax.set_title(layer, color=LAYER_COLORS.get(layer, 'k'), fontsize=11, fontweight='bold')
        ax.set_xlabel('Day')
        ax.set_ylim(0, 1)
        ax.set_xticks(days)

    axes[0].set_ylabel('Proportion of cells')
    axes[-1].legend(fontsize=7, loc='upper right')
    fig.suptitle(f'{animal_id} — Landmark preference trajectory', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f'{animal_id}_B_trajectory.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: B_trajectory')


# ─────────────────────────── Figure C: day1 vs last ──────────────────────────

def plot_day1_vs_last(sessions_data, animal_id, save_path):
    """Side-by-side stacked bars: Day 1 vs last day, per layer."""
    days = _sorted_days(sessions_data)
    if len(days) < 2:
        return
    d1, d_last = days[0], days[-1]
    layers = [l for l in LAYER_ORDER if l in sessions_data[d1]['layers'] or l in sessions_data[d_last]['layers']]

    fig, axes = plt.subplots(1, len(layers), figsize=(3.5 * len(layers), 3.5))
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        for xi, (day, label) in enumerate([(d1, f'Day {d1}'), (d_last, f'Day {d_last}')]):
            ld = sessions_data[day]['layers'].get(layer)
            if ld is None or ld['proportions'] is None:
                continue
            props = ld['proportions']
            bottom = 0.0
            for lm_i, p in enumerate(props):
                ax.bar(xi, p, bottom=bottom, color=LANDMARK_COLORS[lm_i], width=0.5)
                if p > 0.08:
                    ax.text(xi, bottom + p / 2, f'{p:.0%}',
                            ha='center', va='center', fontsize=8, color='white', fontweight='bold')
                bottom += p
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'Day {d1}', f'Day {d_last}'], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(layer, color=LAYER_COLORS.get(layer, 'k'), fontsize=11, fontweight='bold')

    axes[0].set_ylabel('Proportion of cells')
    patches = [mpatches.Patch(color=LANDMARK_COLORS[i], label=f'{LANDMARK_POSITIONS[i]} cm')
               for i in range(N_LM)]
    fig.legend(handles=patches, loc='lower center', ncol=N_LM, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f'{animal_id} — Day 1 vs Day {d_last}', fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(os.path.join(save_path, f'{animal_id}_C_day1_vs_last.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: C_day1_vs_last')


# ─────────────────────────── Figure D: dominant map ─────────────────────────

def plot_dominant_map(sessions_data, animal_id, save_path):
    """Color grid: dominant landmark per layer × day."""
    days = _sorted_days(sessions_data)
    layers = [l for l in LAYER_ORDER if any(l in sessions_data[d]['layers'] for d in days)]

    dom_matrix = np.full((len(layers), len(days)), -1, dtype=int)
    for ri, layer in enumerate(layers):
        for ci, day in enumerate(days):
            ld = sessions_data[day]['layers'].get(layer)
            if ld is not None and ld['proportions'] is not None:
                dom_matrix[ri, ci] = _dominant_index(ld['proportions'])

    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(days)), 1.2 * len(layers) + 1))
    cmap = matplotlib.colors.ListedColormap(LANDMARK_COLORS)

    # Only color valid cells
    img = np.ma.masked_where(dom_matrix < 0, dom_matrix)
    ax.imshow(img, cmap=cmap, vmin=0, vmax=N_LM - 1, aspect='auto')

    ax.set_xticks(range(len(days)))
    ax.set_xticklabels([f'Day {d}' for d in days], fontsize=9)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=10)
    for i, layer in enumerate(layers):
        ax.get_yticklabels()[i].set_color(LAYER_COLORS.get(layer, 'k'))

    patches = [mpatches.Patch(color=LANDMARK_COLORS[i], label=f'{LANDMARK_POSITIONS[i]} cm')
               for i in range(N_LM)]
    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.25, 1), fontsize=8)

    ax.set_title(f'{animal_id} — Dominant landmark per layer × day', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f'{animal_id}_D_dominant_map.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: D_dominant_map')


# ─────────────────────────── Figure E: bias index ────────────────────────────

def plot_bias_index(sessions_data, animal_id, save_path):
    """Landmark bias index (max(p) - 1/N) per layer across days."""
    days = _sorted_days(sessions_data)
    layers = [l for l in LAYER_ORDER if any(l in sessions_data[d]['layers'] for d in days)]

    fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(days)), 4))

    for layer in layers:
        vals = [sessions_data[d]['layers'].get(layer, {}).get('bias_index', np.nan) for d in days]
        vals = np.array(vals, dtype=float)
        valid = ~np.isnan(vals)
        ax.plot(np.array(days)[valid], vals[valid],
                color=LAYER_COLORS.get(layer, 'k'), marker='o', linewidth=2, label=layer)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='Chance level')
    ax.set_xlabel('Day', fontsize=10)
    ax.set_ylabel('Bias Index (max(p) – 1/N)', fontsize=10)
    ax.set_ylim(-0.05, 0.75)
    ax.set_xticks(days)
    ax.legend(fontsize=9)
    ax.set_title(f'{animal_id} — Landmark Bias Index per layer', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f'{animal_id}_E_bias_index.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: E_bias_index')


# ─────────────────────────── Figure F: entropy ───────────────────────────────

def plot_entropy(sessions_data, animal_id, save_path):
    """Normalized Shannon entropy per layer across days."""
    days = _sorted_days(sessions_data)
    layers = [l for l in LAYER_ORDER if any(l in sessions_data[d]['layers'] for d in days)]

    fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(days)), 4))

    for layer in layers:
        vals = [sessions_data[d]['layers'].get(layer, {}).get('entropy', np.nan) for d in days]
        vals = np.array(vals, dtype=float)
        valid = ~np.isnan(vals)
        ax.plot(np.array(days)[valid], vals[valid],
                color=LAYER_COLORS.get(layer, 'k'), marker='o', linewidth=2, label=layer)

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='Uniform (H=1)')
    ax.set_xlabel('Day', fontsize=10)
    ax.set_ylabel('Normalized Entropy', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(days)
    ax.legend(fontsize=9)
    ax.set_title(f'{animal_id} — Landmark Preference Entropy per layer', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f'{animal_id}_F_entropy.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: F_entropy')


# ─────────────────────────── Figure G: early vs late ─────────────────────────

def plot_early_vs_late(sessions_data, animal_id, save_path):
    """Compare first 2 days vs last 2 days bias index per layer."""
    days = _sorted_days(sessions_data)
    if len(days) < 3:
        print('  Skipped: G_early_vs_late (< 3 days)')
        return

    early_days = days[:2]
    late_days  = days[-2:]
    layers = [l for l in LAYER_ORDER if any(l in sessions_data[d]['layers'] for d in days)]

    early_bias = {layer: np.nanmean([sessions_data[d]['layers'].get(layer, {}).get('bias_index', np.nan)
                                      for d in early_days]) for layer in layers}
    late_bias  = {layer: np.nanmean([sessions_data[d]['layers'].get(layer, {}).get('bias_index', np.nan)
                                      for d in late_days]) for layer in layers}

    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(5, 1.5 * len(layers)), 4))
    bars_e = ax.bar(x - width / 2,
                    [early_bias[l] for l in layers],
                    width, label=f'Early (D{early_days[0]}–D{early_days[-1]})',
                    color=[LAYER_COLORS.get(l, 'k') for l in layers], alpha=0.5)
    bars_l = ax.bar(x + width / 2,
                    [late_bias[l] for l in layers],
                    width, label=f'Late (D{late_days[0]}–D{late_days[-1]})',
                    color=[LAYER_COLORS.get(l, 'k') for l in layers], alpha=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=10)
    ax.set_ylabel('Mean Bias Index', fontsize=10)
    ax.set_ylim(0, 0.75)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.legend(fontsize=9)
    ax.set_title(f'{animal_id} — Early vs Late landmark bias', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f'{animal_id}_G_early_vs_late.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: G_early_vs_late')


# ─────────────────────────── Figure H: within-session dynamics ───────────────

def plot_within_session_dynamics(sessions_data, animal_id, save_path):
    """
    Intra-session dynamics: preference_by_block [N_blocks, N_LM] per layer,
    one subplot row per day.
    """
    days = _sorted_days(sessions_data)
    layers = [l for l in LAYER_ORDER
              if any(sessions_data[d].get('dynamics') is not None and
                     l in sessions_data[d]['dynamics']['preference_by_block']
                     for d in days)]

    if not layers:
        print('  Skipped: H_within_session (no dynamics data)')
        return

    n_rows = len(days)
    n_cols = len(layers)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.5 * n_cols, 2.5 * n_rows),
                              sharex='col', sharey=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for ri, day in enumerate(days):
        dyn = sessions_data[day].get('dynamics')
        for ci, layer in enumerate(layers):
            ax = axes[ri, ci]
            if dyn is None or layer not in dyn['preference_by_block']:
                ax.set_visible(False)
                continue
            block_props = dyn['preference_by_block'][layer]   # [N_blocks, N_LM]
            n_blocks = block_props.shape[0]
            block_idx = np.arange(1, n_blocks + 1)
            for lm_i in range(N_LM):
                ax.plot(block_idx, block_props[:, lm_i],
                        color=LANDMARK_COLORS[lm_i], marker='o', markersize=3,
                        linewidth=1.4, label=f'{LANDMARK_POSITIONS[lm_i]} cm')
            ax.axhline(1 / N_LM, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
            ax.set_ylim(0, 1)
            if ri == 0:
                ax.set_title(layer, color=LAYER_COLORS.get(layer, 'k'), fontsize=9, fontweight='bold')
            if ci == 0:
                ax.set_ylabel(f'Day {day}', fontsize=8)
            if ri == n_rows - 1:
                ax.set_xlabel('Block', fontsize=8)

    axes[0, -1].legend(fontsize=6, loc='upper right')
    fig.suptitle(f'{animal_id} — Within-session dynamics', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f'{animal_id}_H_within_session.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: H_within_session')


# ─────────────────────────── Figure I: summary ───────────────────────────────

def plot_summary(sessions_data, animal_id, save_path):
    """
    Summary panel: bias index (top) + entropy (mid) trajectories,
    plus JSD day-over-day stability (bottom).
    """
    days = _sorted_days(sessions_data)
    layers = [l for l in LAYER_ORDER if any(l in sessions_data[d]['layers'] for d in days)]

    fig = plt.figure(figsize=(max(8, 1.6 * len(days)), 10))
    gs  = gridspec.GridSpec(3, 1, hspace=0.45)

    # --- top: bias index ---
    ax_bias = fig.add_subplot(gs[0])
    for layer in layers:
        vals = [sessions_data[d]['layers'].get(layer, {}).get('bias_index', np.nan) for d in days]
        vals = np.array(vals, dtype=float)
        valid = ~np.isnan(vals)
        ax_bias.plot(np.array(days)[valid], vals[valid],
                     color=LAYER_COLORS.get(layer, 'k'), marker='o', linewidth=2, label=layer)
    ax_bias.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax_bias.set_ylabel('Bias Index', fontsize=9)
    ax_bias.set_ylim(-0.05, 0.75)
    ax_bias.set_xticks(days)
    ax_bias.legend(fontsize=8, loc='upper left')
    ax_bias.set_title('Landmark Bias Index', fontsize=10)

    # --- mid: entropy ---
    ax_ent = fig.add_subplot(gs[1])
    for layer in layers:
        vals = [sessions_data[d]['layers'].get(layer, {}).get('entropy', np.nan) for d in days]
        vals = np.array(vals, dtype=float)
        valid = ~np.isnan(vals)
        ax_ent.plot(np.array(days)[valid], vals[valid],
                    color=LAYER_COLORS.get(layer, 'k'), marker='o', linewidth=2, label=layer)
    ax_ent.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='Uniform')
    ax_ent.set_ylabel('Norm. Entropy', fontsize=9)
    ax_ent.set_ylim(0, 1.1)
    ax_ent.set_xticks(days)
    ax_ent.legend(fontsize=8, loc='lower left')
    ax_ent.set_title('Normalized Shannon Entropy', fontsize=10)

    # --- bottom: JSD stability (mean across layers) ---
    ax_jsd = fig.add_subplot(gs[2])
    jsd_all = []
    for layer in layers:
        mat = _prop_matrix(sessions_data, layer)
        valid_rows = ~np.any(np.isnan(mat), axis=1)
        if np.sum(valid_rows) >= 2:
            jsd_vals = jsd_adjacent(mat[valid_rows])
            # map back to adjacent day pairs
            valid_day_indices = np.where(valid_rows)[0]
            jsd_all.append(jsd_vals)
            mid_days = [(days[valid_day_indices[i]] + days[valid_day_indices[i + 1]]) / 2
                        for i in range(len(jsd_vals))]
            ax_jsd.plot(mid_days, jsd_vals,
                        color=LAYER_COLORS.get(layer, 'k'), marker='s', linewidth=1.5,
                        markersize=5, label=layer, alpha=0.7)

    ax_jsd.set_ylabel('JSD (adjacent days)', fontsize=9)
    ax_jsd.set_ylim(0, 1)
    ax_jsd.set_xlabel('Day (midpoint)', fontsize=9)
    ax_jsd.legend(fontsize=8)
    ax_jsd.set_title('Day-over-day landmark preference stability (lower = more stable)', fontsize=10)

    fig.suptitle(f'{animal_id} — Landmark Preference Summary', fontsize=13, fontweight='bold')
    fig.savefig(os.path.join(save_path, f'{animal_id}_I_summary.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: I_summary')


# ─────────────────────────── main analysis runner ────────────────────────────

def run_within_animal_analysis(animal_dir, animal_id=None):
    """
    Load landmark h5 files and produce all figures for one animal.
    Figures saved to {animal_dir}/LandmarkPreference/.
    """
    print(f'\n{"="*70}')
    print(f'WITHIN-ANIMAL LANDMARK PREFERENCE ANALYSIS')
    if animal_id:
        print(f'Animal: {animal_id}')
    print(f'Directory: {animal_dir}')
    print(f'{"="*70}')

    sessions_data, detected_id = load_animal_landmark_data(animal_dir)
    if animal_id is None:
        animal_id = detected_id or os.path.basename(animal_dir)

    if not sessions_data:
        print('  No sessions loaded — skipping.')
        return

    save_path = os.path.join(animal_dir, 'LandmarkPreference')
    os.makedirs(save_path, exist_ok=True)
    print(f'\n  Saving figures to: {save_path}')
    print(f'  Days found: {_sorted_days(sessions_data)}\n')

    plot_stacked_bars(sessions_data, animal_id, save_path)
    plot_trajectory(sessions_data, animal_id, save_path)
    plot_day1_vs_last(sessions_data, animal_id, save_path)
    plot_dominant_map(sessions_data, animal_id, save_path)
    plot_bias_index(sessions_data, animal_id, save_path)
    plot_entropy(sessions_data, animal_id, save_path)
    plot_early_vs_late(sessions_data, animal_id, save_path)
    plot_within_session_dynamics(sessions_data, animal_id, save_path)
    plot_summary(sessions_data, animal_id, save_path)

    print(f'\n  Done — {animal_id}')
    return sessions_data


# ─────────────────────────── entry point ────────────────────────────────────

if __name__ == '__main__':
    # Set ANIMAL to a specific ID (e.g. 'JSY054') or None to run all animals
    ANIMAL = None

    if ANIMAL is not None:
        animal_dir = ANIMAL_DIRS[ANIMAL]
        run_within_animal_analysis(animal_dir, animal_id=ANIMAL)
    else:
        for aid, adir in ANIMAL_DIRS.items():
            if os.path.isdir(adir):
                run_within_animal_analysis(adir, animal_id=aid)
            else:
                print(f'  Skipping {aid} — directory not found: {adir}')
