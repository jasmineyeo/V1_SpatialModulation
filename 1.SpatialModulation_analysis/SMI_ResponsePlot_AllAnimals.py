"""
SMI_ResponsePlot_AllAnimals.py

Generates spatial response heatmaps (tuning curves sorted by preferred position).
Loads norm_spatial_activity from *_preproc*.h5 and layer/cell assignments
from *_smi_results.h5.

Outputs (saved to session folder or summary folder):
  A. Per-session: 4-panel figure (one panel per layer) — {animal}_{day}_response_session.png
  B. Per-animal:  4-layer × N-days grid               — {animal}_response_grid.png
  C. Pooled:      4-layer × 7-days grid (all animals)  — across_animals_response_grid.png
  D. Per-day:     all layers stacked (all animals)      — Day{N}_response_all_layers.png

JSY, 2025
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import glob
import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import h5py

# Parula colormap (MATLAB-compatible, perceptually uniform blue→cyan→green→yellow)
_PARULA_COLORS = [
    (0.2422, 0.1504, 0.6603),
    (0.2108, 0.3706, 0.9717),
    (0.0196, 0.5804, 0.8745),
    (0.0863, 0.6510, 0.7490),
    (0.1961, 0.6980, 0.6039),
    (0.3647, 0.7412, 0.5176),
    (0.6275, 0.7647, 0.3843),
    (0.8510, 0.7882, 0.1961),
    (0.9686, 0.8235, 0.0667),
    (0.9765, 0.9843, 0.0510),
]
PARULA = mcolors.LinearSegmentedColormap.from_list('parula', _PARULA_COLORS)

# ============================================================================
# CONFIG
# ============================================================================

LAYER_ORDER = ['L2/3', 'L4', 'L5', 'L6']
LAYER_COLORS = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
ALL_DAYS = list(range(1, 8))

ANIMAL_DIRS = {
    'JSY040': r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging',
    'JSY041': r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging',
    'JSY044': r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging',
    'JSY051': r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging',
    'JSY052': r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging',
    'JSY054': r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging',
    'JSY055': r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging',
}

SESSION_DIRS = [
    # JSY040
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250620_JSY_JSY040_SpatialModulation_Day1_V1Prism\TSeries-06202025-1515-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250622_JSY_JSY040_SpatialModulation_Day3_V1Prism\TSeries-06222025-1550-001',
    # JSY041
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250616_JSY_JSY041_SpatialModulation_Day1_V1Prism\TSeries-06162025-1521-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250618_JSY_JSY041_SpatialModulation_Day3_V1Prism\TSeries-06182025-1641-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250620_JSY_JSY041_SpatialModulation_Day5_V1Prism\TSeries-06202025-1515-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250622_JSY_JSY041_SpatialModulation_Day7_V1Prism\TSeries-06222025-1550-001',
    # JSY044
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1\TSeries-09062025-1308-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250907_JSY_JSY044_SpaitalModulation_Day2\TSeries-09072025-1257-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3\TSeries-09082025-1540-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250909_JSY_JSY044_SpatialModulation_Day4\TSeries-09092025-1256-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250910_JSY_JSY044_SpatialModulation_Day5\TSeries-09102025-1340-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-001',
    # JSY051
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251101_JSY_JSY051_SpMod_Day1\TSeries-11012025-1725-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251102_JSY_JSY051_SpMod_Day2\TSeries-11022025-1642-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251103_JSY_JSY051_SpMod_Day3\TSeries-11032025-1715-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251104_JSY_JSY051_SpMod_Day4\TSeries-11042025-1418-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251105_JSY_JSY051_SpMod_Day5\TSeries-11052025-1512-002',
    # JSY052
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251009_JSY_JSY052_SpatialModulation_Day1\TSeries-10092025-1542-002',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251010_JSY_JSY052_SpatialModulation_Day2\TSeries-10102025-0916-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251011_JSY_JSY052_SpatialModulation_Day3\TSeries-10112025-1441-002',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251012_JSY_JSY052_SpatialModulation_Day4\TSeries-10122025-1212-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251013_JSY_JSY052_SpatialModulation_Day5\TSeries-10132025-1236-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251014_JSY_JSY052_SpatialModulation_Day6\TSeries-10142025-1647-003',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251015_JSY_JSY052_SpatialModulation_Day7\TSeries-10152025-1103-001',
    # JSY054
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251030_JSY_JSY054_SpMod_Day1\TSeries-10302025-1512-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251031_JSY_JSY054_SpMod_Day2\TSeries-10312025-1751-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251101_JSY_JSY054_SpMod_Day3\TSeries-11012025-1725-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251102_JSY_JSY054_SpMod_Day4\TSeries-11022025-1642-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251103_JSY_JSY054_SpMod_Day5\TSeries-11032025-1715-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251104_JSY_JSY054_SpMod_Day6\TSeries-11042025-1418-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251105_JSY_JSY054_SpMod_Day7\TSeries-11052025-1512-001',
    # JSY055
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251205_JSY_JSY055_SpatialModulation_Day1\TSeries-12052025-1740-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251206_JSY_JSY055_SpatialModulation_Day2\TSeries-12062025-1810-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251207_JSY_JSY055_SpatialModulation_Day3\TSeries-12072025-1825-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251208_JSY_JSY055_SpatialModulation_Day4\TSeries-12082025-1633-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251209_JSY_JSY055_SpatialModualtion_Day5\TSeries-12092025-2000-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251210_JSY_JSY055_SpatialModulation_Day6\TSeries-12102025-1702-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251211_JSY_JSY055_SpatialModulation_Day7\TSeries-12112025-1631-001',
]


# ============================================================================
# DATA LOADING
# ============================================================================

def _parse_animal_day_from_path(session_dir):
    """Extract animal ID and day number from the session folder path."""
    import re
    # Parent folder name e.g. '250906_JSY_JSY044_SpatialModulation_Day1'
    parent = os.path.basename(os.path.dirname(session_dir))
    m_animal = re.search(r'(JSY\d+)', parent)
    m_day = re.search(r'[Dd]ay(\d+)', parent)
    animal = m_animal.group(1) if m_animal else 'unknown'
    day_num = int(m_day.group(1)) if m_day else 0
    day_str = f'Day{day_num}' if day_num else 'unknown'
    return animal, day_str, day_num


def load_session_data(session_dir):
    """
    Load one session's tuning curves and layer assignments.
    Returns dict: {animal, day, day_num, session_dir, bin_centers, layers}
      layers[layer_name] = {tuning_norm (n_cells, n_bins), preferred_positions, smi}
    """
    smi_files = glob.glob(os.path.join(session_dir, '*_smi_results.h5'))
    preproc_files = glob.glob(os.path.join(session_dir, '*preproc*.h5'))

    if not smi_files:
        raise FileNotFoundError('No *_smi_results.h5 found')
    if not preproc_files:
        raise FileNotFoundError('No *_preproc*.h5 found')

    # Always extract animal/day from path (h5 attrs may be absent)
    animal, day_str, day_num = _parse_animal_day_from_path(session_dir)

    with h5py.File(preproc_files[0], 'r') as f:
        norm_act = f['norm_spatial_activity'][:]   # (N_cells, N_laps, N_bins)
        bin_centers = f['bin_centers'][:]

    mean_tuning = np.nanmean(norm_act, axis=1)     # (N_cells, N_bins)

    with h5py.File(smi_files[0], 'r') as f:
        layers = {}
        for lk in f['layer_smi'].keys():
            layer_name = lk.replace('_', '/')
            lg = f['layer_smi'][lk]
            rv = lg['reliable_valid_cells'][:].astype(int)
            pp = lg['preferred_positions'][:]
            smi = lg['SMI'][:]
            if len(rv) == 0:
                continue
            tuning = mean_tuning[rv, :]
            row_max = tuning.max(axis=1, keepdims=True)
            row_max[row_max == 0] = 1
            layers[layer_name] = {
                'tuning_norm': tuning / row_max,
                'preferred_positions': pp,
                'smi': smi,
            }

    return {
        'animal': animal, 'day': day_str, 'day_num': day_num,
        'session_dir': session_dir,
        'bin_centers': bin_centers, 'layers': layers,
    }


def load_all_sessions():
    """
    Load all sessions. Returns:
      by_session: list of session_data dicts (for per-session plots)
      by_animal:  {animal: {day_num: session_data}}
    """
    by_session = []
    by_animal = {}

    for session_dir in SESSION_DIRS:
        if not os.path.isdir(session_dir):
            continue
        if not glob.glob(os.path.join(session_dir, '*_smi_results.h5')):
            continue
        try:
            sd = load_session_data(session_dir)
            by_session.append(sd)
            by_animal.setdefault(sd['animal'], {})[sd['day_num']] = sd
            print(f'  Loaded {sd["animal"]} {sd["day"]}')
        except Exception as e:
            print(f'  ERROR {os.path.basename(session_dir)}: {e}')

    return by_session, by_animal


# ============================================================================
# SHARED PLOTTING HELPER
# ============================================================================

def _draw_heatmap(ax, tuning_norm, preferred_positions, bin_centers,
                  title='', show_pref_strip=True):
    """
    Draw a sorted tuning-curve heatmap on ax.
    tuning_norm: (n_cells, n_bins), already row-normalised.
    """
    if tuning_norm is None or len(tuning_norm) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=8, color='gray')
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    order = np.argsort(preferred_positions)
    tuning_sorted = tuning_norm[order]
    pp_sorted = preferred_positions[order]
    n_cells = len(order)
    pos_min, pos_max = bin_centers[0], bin_centers[-1]

    ax.imshow(tuning_sorted, aspect='auto', cmap=PARULA, vmin=0, vmax=1,
              extent=[pos_min, pos_max, n_cells, 0], interpolation='nearest')

    if show_pref_strip:
        pp_norm = (pp_sorted - pos_min) / max(pos_max - pos_min, 1)
        strip = pp_norm[:, np.newaxis]
        ax_s = ax.inset_axes([1.01, 0, 0.04, 1], transform=ax.transAxes)
        ax_s.imshow(strip, aspect='auto', cmap='twilight', vmin=0, vmax=1,
                    extent=[0, 1, n_cells, 0], interpolation='nearest')
        ax_s.set_xticks([])
        ax_s.set_yticks([])

    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.set_xlim(pos_min, pos_max)
    ax.set_xticks([])
    ax.set_yticks([])


def _pool_layer_data(sessions):
    """
    Pool tuning_norm and preferred_positions across a list of session_data dicts.
    Returns {layer: {tuning_norm, preferred_positions, bin_centers}} or None per layer.
    """
    pooled = {}
    bin_centers_ref = None
    for sd in sessions:
        if bin_centers_ref is None:
            bin_centers_ref = sd['bin_centers']
        for layer, ld in sd['layers'].items():
            if layer not in pooled:
                pooled[layer] = {'tuning_norm': [], 'preferred_positions': []}
            pooled[layer]['tuning_norm'].append(ld['tuning_norm'])
            pooled[layer]['preferred_positions'].append(ld['preferred_positions'])

    result = {}
    for layer, data in pooled.items():
        result[layer] = {
            'tuning_norm': np.concatenate(data['tuning_norm'], axis=0),
            'preferred_positions': np.concatenate(data['preferred_positions'], axis=0),
            'bin_centers': bin_centers_ref,
        }
    return result, bin_centers_ref


# ============================================================================
# FIGURE A: PER-SESSION (4 layers, one session)
# ============================================================================

def plot_session_response(session_data, save_path=None, skip_existing=True):
    animal, day = session_data['animal'], session_data['day']
    fname = f'{animal}_{day}_response_session.png'
    out = os.path.join(save_path, fname) if save_path else None
    if out and skip_existing and os.path.exists(out):
        print(f'  Skipped (exists): {fname}')
        return

    layers_present = [l for l in LAYER_ORDER if l in session_data['layers']]
    n = max(len(layers_present), 1)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 7), constrained_layout=True)
    if n == 1:
        axes = [axes]

    bc = session_data['bin_centers']
    for ax, layer in zip(axes, layers_present):
        ld = session_data['layers'][layer]
        color = LAYER_COLORS.get(layer, 'black')
        _draw_heatmap(ax, ld['tuning_norm'], ld['preferred_positions'], bc, title=layer)
        ax.set_title(layer, color=color, fontsize=11, fontweight='bold')

    fig.supxlabel('VR position (cm)', fontsize=13, fontweight='bold')
    fig.suptitle(f'{animal} — {day}: Spatial tuning (reliable cells, sorted by pref. pos.)',
                 fontsize=12, fontweight='bold')
    if out:
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved: {fname}')
    plt.close(fig)


# ============================================================================
# FIGURE B: PER-ANIMAL GRID (4 layers × N days)
# ============================================================================

def plot_animal_grid(animal_days, animal_id, save_dir, skip_existing=True):
    """
    Rows = layers (L2/3, L4, L5, L6). Columns = days sorted.
    """
    fname = f'{animal_id}_response_grid.png'
    out = os.path.join(save_dir, fname)
    if skip_existing and os.path.exists(out):
        print(f'  Skipped (exists): {fname}')
        return

    days = sorted(animal_days.keys())
    n_days = len(days)
    n_layers = len(LAYER_ORDER)

    fig, axes = plt.subplots(n_layers, n_days,
                             figsize=(3.5 * n_days, 3 * n_layers),
                             squeeze=False)

    for col, day in enumerate(days):
        sd = animal_days[day]
        bc = sd['bin_centers']
        for row, layer in enumerate(LAYER_ORDER):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(f'Day {day}', fontsize=9, fontweight='bold')
            if col == 0:
                color = LAYER_COLORS.get(layer, 'black')
                ax.set_ylabel(layer, fontsize=9, fontweight='bold', color=color)

            if layer in sd['layers']:
                ld = sd['layers'][layer]
                _draw_heatmap(ax, ld['tuning_norm'], ld['preferred_positions'],
                              bc, show_pref_strip=(col == n_days - 1))
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=7, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])

    fig.supxlabel('VR position (cm)', fontsize=13, fontweight='bold')
    fig.suptitle(f'{animal_id}: Spatial Tuning Grid (4 layers × {n_days} days)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  Saved: {fname}')
    plt.close(fig)


# ============================================================================
# FIGURE C: POOLED ACROSS ALL ANIMALS GRID (4 layers × 7 days)
# ============================================================================

def plot_pooled_grid(by_animal, save_dir, skip_existing=True):
    """
    Rows = layers. Columns = days 1–7.
    Each cell pools all reliable cells from all animals for that (layer, day).
    """
    fname = 'across_animals_response_grid.png'
    out = os.path.join(save_dir, fname)
    if skip_existing and os.path.exists(out):
        print(f'  Skipped (exists): {fname}')
        return

    n_layers = len(LAYER_ORDER)
    n_days = len(ALL_DAYS)
    fig, axes = plt.subplots(n_layers, n_days,
                             figsize=(3.5 * n_days, 3 * n_layers),
                             squeeze=False)

    for col, day in enumerate(ALL_DAYS):
        # Collect all sessions for this day across all animals
        sessions_this_day = [
            animal_days[day]
            for animal_days in by_animal.values()
            if day in animal_days
        ]
        n_animals = len(sessions_this_day)

        for row, layer in enumerate(LAYER_ORDER):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(f'Day {day}\n({n_animals} mice)', fontsize=8, fontweight='bold')
            if col == 0:
                color = LAYER_COLORS.get(layer, 'black')
                ax.set_ylabel(layer, fontsize=9, fontweight='bold', color=color)
            # Pool tuning curves
            tunings, pps, bc = [], [], None
            for sd in sessions_this_day:
                if bc is None:
                    bc = sd['bin_centers']
                if layer in sd['layers']:
                    tunings.append(sd['layers'][layer]['tuning_norm'])
                    pps.append(sd['layers'][layer]['preferred_positions'])

            if tunings and bc is not None:
                _draw_heatmap(
                    ax,
                    np.concatenate(tunings, axis=0),
                    np.concatenate(pps, axis=0),
                    bc,
                    show_pref_strip=(col == n_days - 1),
                )
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=7, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])

    fig.supxlabel('VR position (cm)', fontsize=13, fontweight='bold')
    fig.suptitle('Across Animals: Spatial Tuning Grid (4 layers × 7 days, pooled cells)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  Saved: {fname}')
    plt.close(fig)


# ============================================================================
# FIGURE D: PER-DAY ALL-LAYERS STACKED (one figure per day, all animals pooled)
# ============================================================================

def plot_per_day_all_layers(by_animal, save_dir, skip_existing=True):
    """
    For each day: one figure with 4 subplots (one per layer), all animals pooled.
    Shows all 4 layers as a vertically stacked multi-panel heatmap per day.
    7 figures total (one per day).
    """
    for day in ALL_DAYS:
        fname = f'Day{day}_response_all_layers.png'
        out = os.path.join(save_dir, fname)
        if skip_existing and os.path.exists(out):
            print(f'  Skipped (exists): {fname}')
            continue

        sessions_this_day = [
            animal_days[day]
            for animal_days in by_animal.values()
            if day in animal_days
        ]
        n_animals = len(sessions_this_day)
        if n_animals == 0:
            print(f'  Day {day}: no data, skipping')
            continue

        # Calculate per-layer heights proportional to cell count for the gridspec
        layer_ncells = []
        layer_tunings = []
        layer_pps = []
        bc = None
        for layer in LAYER_ORDER:
            tunings, pps = [], []
            for sd in sessions_this_day:
                if bc is None:
                    bc = sd['bin_centers']
                if layer in sd['layers']:
                    tunings.append(sd['layers'][layer]['tuning_norm'])
                    pps.append(sd['layers'][layer]['preferred_positions'])
            if tunings:
                t = np.concatenate(tunings, axis=0)
                p = np.concatenate(pps, axis=0)
            else:
                t, p = None, None
            layer_tunings.append(t)
            layer_pps.append(p)
            layer_ncells.append(len(p) if p is not None else 1)

        # Equal square panels: figure height = width × number of layers
        n_layers = len(LAYER_ORDER)
        panel_size = 8  # inches per panel (square)
        fig = plt.figure(figsize=(panel_size, panel_size * n_layers))
        gs = gridspec.GridSpec(n_layers, 1, hspace=0.05)

        for row, (layer, t, p, nc) in enumerate(
                zip(LAYER_ORDER, layer_tunings, layer_pps, layer_ncells)):
            ax = fig.add_subplot(gs[row])
            color = LAYER_COLORS.get(layer, 'black')

            if t is not None and len(t) > 0 and bc is not None:
                order = np.argsort(p)
                pos_min, pos_max = bc[0], bc[-1]
                ax.imshow(t[order], aspect='auto', cmap=PARULA, vmin=0, vmax=1,
                          extent=[pos_min, pos_max, nc, 0], interpolation='nearest')

                # Pref-pos strip on right
                pp_norm = (p[order] - pos_min) / max(pos_max - pos_min, 1)
                strip = pp_norm[:, np.newaxis]
                ax_s = ax.inset_axes([1.01, 0, 0.025, 1], transform=ax.transAxes)
                ax_s.imshow(strip, aspect='auto', cmap='twilight', vmin=0, vmax=1,
                            extent=[0, 1, nc, 0], interpolation='nearest')
                ax_s.set_xticks([])
                ax_s.set_yticks([])

                ax.set_xlim(pos_min, pos_max)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, color='gray')

            ax.set_ylabel(layer, fontsize=11, fontweight='bold',
                          color=color, rotation=0, labelpad=40, va='center')
            ax.set_yticks([])
            ax.set_xticks([])

        fig.supxlabel('VR position (cm)', fontsize=13, fontweight='bold')
        fig.suptitle(f'Day {day}: All Layers — Spatial Tuning '
                     f'(pooled {n_animals} animals, sorted by pref. pos.)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved: {fname}')
        plt.close(fig)


# ============================================================================
# FIGURE E: PER-DAY ALL-CELLS (all layers + all animals pooled, one heatmap)
# ============================================================================

def plot_per_day_all_cells(by_animal, save_dir, skip_existing=True):
    """
    For each day: one figure with a single heatmap of ALL reliable cells from ALL
    layers and ALL animals pooled together, sorted by preferred position.
    7 figures total — Day{N}_response_all_cells.png.
    """
    for day in ALL_DAYS:
        fname = f'Day{day}_response_all_cells.png'
        out = os.path.join(save_dir, fname)
        if skip_existing and os.path.exists(out):
            print(f'  Skipped (exists): {fname}')
            continue

        sessions_this_day = [
            animal_days[day]
            for animal_days in by_animal.values()
            if day in animal_days
        ]
        n_animals = len(sessions_this_day)
        if n_animals == 0:
            print(f'  Day {day}: no data, skipping')
            continue

        # Pool ALL cells across all layers and all animals
        all_tunings, all_pps = [], []
        bc = None
        for sd in sessions_this_day:
            if bc is None:
                bc = sd['bin_centers']
            for layer in LAYER_ORDER:
                if layer in sd['layers']:
                    all_tunings.append(sd['layers'][layer]['tuning_norm'])
                    all_pps.append(sd['layers'][layer]['preferred_positions'])

        if not all_tunings or bc is None:
            print(f'  Day {day}: no data, skipping')
            continue

        tuning_all = np.concatenate(all_tunings, axis=0)
        pp_all = np.concatenate(all_pps, axis=0)
        n_cells = len(pp_all)

        order = np.argsort(pp_all)
        tuning_sorted = tuning_all[order]
        pp_sorted = pp_all[order]
        pos_min, pos_max = bc[0], bc[-1]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(tuning_sorted, aspect='auto', cmap=PARULA, vmin=0, vmax=1,
                  extent=[pos_min, pos_max, n_cells, 0], interpolation='nearest')

        # Preferred-position strip
        pp_norm = (pp_sorted - pos_min) / max(pos_max - pos_min, 1)
        strip = pp_norm[:, np.newaxis]
        ax_s = ax.inset_axes([1.01, 0, 0.025, 1], transform=ax.transAxes)
        ax_s.imshow(strip, aspect='auto', cmap='twilight', vmin=0, vmax=1,
                    extent=[0, 1, n_cells, 0], interpolation='nearest')
        ax_s.set_xticks([])
        ax_s.set_yticks([])

        ax.set_xlim(pos_min, pos_max)
        ax.set_xticks([])
        ax.set_yticks([])

        fig.supxlabel('VR position (cm)', fontsize=13, fontweight='bold')
        fig.suptitle(f'Day {day}: All Cells (all layers pooled) — '
                     f'{n_cells} cells from {n_animals} animals',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved: {fname}')
        plt.close(fig)


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_all(skip_existing=True):
    summary_dir = r'D:\V1_SpatialModulation\2p\V1_prism\response_plots'
    os.makedirs(summary_dir, exist_ok=True)

    print('\n' + '=' * 80)
    print(' SPATIAL RESPONSE PLOTS — ALL SESSIONS & SUMMARY FIGURES')
    print('=' * 80)

    print('\nLoading all sessions...')
    by_session, by_animal = load_all_sessions()
    print(f'  Loaded {len(by_session)} sessions from {len(by_animal)} animals\n')

    # --- Figure A: per session ---
    print('--- Figure A: Per-session response plots ---')
    for sd in by_session:
        plot_session_response(sd, save_path=sd['session_dir'], skip_existing=skip_existing)

    # --- Figure B: per animal grid ---
    print('\n--- Figure B: Per-animal grid (4 layers × N days) ---')
    for animal, animal_days in by_animal.items():
        animal_save = ANIMAL_DIRS.get(animal, summary_dir)
        os.makedirs(animal_save, exist_ok=True)
        plot_animal_grid(animal_days, animal, animal_save, skip_existing=skip_existing)

    # --- Figure C: pooled grid ---
    print('\n--- Figure C: Pooled across-animals grid (4 layers × 7 days) ---')
    plot_pooled_grid(by_animal, summary_dir, skip_existing=skip_existing)

    # --- Figure D: per-day all layers ---
    print('\n--- Figure D: Per-day all-layers stacked (7 figures) ---')
    plot_per_day_all_layers(by_animal, summary_dir, skip_existing=skip_existing)

    # --- Figure E: per-day all cells pooled ---
    print('\n--- Figure E: Per-day all-cells pooled (7 figures) ---')
    plot_per_day_all_cells(by_animal, summary_dir, skip_existing=skip_existing)

    print(f'\nSummary figures saved to: {summary_dir}')
    print('Done.')


if __name__ == '__main__':
    run_all(skip_existing=True)
