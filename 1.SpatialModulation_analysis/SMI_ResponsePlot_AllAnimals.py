"""
SMI_ResponsePlot_AllAnimals.py

Generates spatial response heatmaps (tuning curves sorted by preferred position).
Loads norm_spatial_activity from *_preproc*.h5 and layer/cell assignments
from *_smi_results.h5.

Outputs (saved to session folder or summary folder):
  A. Per-session: 4-panel figure (one panel per layer) — {animal}_{day}_response_session.png
  B. Per-animal:  4-layer × N-days grid               — {animal}_response_grid.png
  B2.Per-animal:  1-row × N-days strip (all layers)   — {animal}_all_days_strip.png
  C. Pooled:      4-layer × 7-days grid (all animals)  — across_animals_response_grid.png
  D. Per-day:     all layers stacked (all animals)      — Day{N}_response_all_layers.png
  F. All-days strip: 1 row × N_days (all animals+layers pooled) — all_days_response_strip.png

JSY, 2025
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import glob
import traceback
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib import rcParams
rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
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
TRIAL_BIN_SIZE = 10   # laps per trial-bin for within-session plots

# Cell selection mode:
#   False → use reliable_valid_cells from *_smi_results.h5
#           (valid_cells AND reliable_cells, then intersected with layer)
#   True  → use combined_reliable from *_preproc*.h5
#           (lap-to-lap reliability only, intersected with layer cell_indices)
USE_COMBINED_RELIABLE = True

ANIMAL_DIRS = {
    'JSY040': r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging',
    # 'JSY041': r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging',
    # 'JSY044': r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging',
    'JSY051': r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging',
    'JSY052': r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging',
    'JSY054': r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging',
    'JSY055': r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging',
}

SESSION_DIRS = [
    # JSY040
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250620_JSY_JSY040_SpatialModulation_Day1_V1Prism\TSeries-06202025-1515-001',
    r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250622_JSY_JSY040_SpatialModulation_Day3_V1Prism\TSeries-06222025-1550-001',
    
    # # JSY041
    # r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250616_JSY_JSY041_SpatialModulation_Day1_V1Prism\TSeries-06162025-1521-001',
    # r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250618_JSY_JSY041_SpatialModulation_Day3_V1Prism\TSeries-06182025-1641-001',
    # r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250620_JSY_JSY041_SpatialModulation_Day5_V1Prism\TSeries-06202025-1515-001',
    # r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250622_JSY_JSY041_SpatialModulation_Day7_V1Prism\TSeries-06222025-1550-001',
    
    # # JSY044
    # # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250811_JSY_JSY044_SpatialModulation_Day1\TSeries-08112025-1505-001'
    # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1\TSeries-09062025-1308-001',
    # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250907_JSY_JSY044_SpaitalModulation_Day2\TSeries-09072025-1257-001',
    # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3\TSeries-09082025-1540-001',
    # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250909_JSY_JSY044_SpatialModulation_Day4\TSeries-09092025-1256-001',
    # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250910_JSY_JSY044_SpatialModulation_Day5\TSeries-09102025-1340-001',
    # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-001',
    # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-001',
    
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
        combined_reliable = f['combined_reliable'][:].astype(bool) if 'combined_reliable' in f else None

    mean_tuning = np.nanmean(norm_act, axis=1)     # (N_cells, N_bins)

    with h5py.File(smi_files[0], 'r') as f:
        layers = {}
        for lk in f['layer_smi'].keys():
            layer_name = lk.replace('_', '/')
            lg = f['layer_smi'][lk]

            if USE_COMBINED_RELIABLE:
                if combined_reliable is None:
                    continue
                layer_cell_indices = lg['cell_indices'][:].astype(int)
                rv = layer_cell_indices[combined_reliable[layer_cell_indices]]
            else:
                if 'reliable_valid_cells' not in lg:
                    continue
                rv = lg['reliable_valid_cells'][:].astype(int)

            if len(rv) == 0:
                continue

            pp = lg['preferred_positions'][:] if 'preferred_positions' in lg else None
            smi = lg['SMI'][:] if 'SMI' in lg else np.full(len(rv), np.nan)

            if USE_COMBINED_RELIABLE or pp is None:
                # preferred_positions in the h5 are indexed to reliable_valid_cells;
                # when using combined_reliable we recompute from the tuning curve peak
                tuning = mean_tuning[rv, :]
                row_max = tuning.max(axis=1, keepdims=True)
                row_max[row_max == 0] = 1
                tuning_norm = tuning / row_max
                pp = bin_centers[np.argmax(tuning_norm, axis=1)]
            else:
                tuning = mean_tuning[rv, :]
                row_max = tuning.max(axis=1, keepdims=True)
                row_max[row_max == 0] = 1
                tuning_norm = tuning / row_max

            layers[layer_name] = {
                'tuning_norm': tuning_norm,
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
                  title='', show_pref_strip=True,
                  show_ncells=False, show_ncells_ylabel=True, show_xaxis=False):
    """
    Draw a sorted tuning-curve heatmap on ax.
    tuning_norm        : (n_cells, n_bins), already row-normalised.
    show_ncells        : if True, show [0, n_cells] ticks on y-axis.
    show_ncells_ylabel : if True (and show_ncells), also show '# cells' label text.
                         Set False on inner columns to save horizontal space.
    show_xaxis         : if True, show position ticks on x-axis (bottom-row panels).
    """
    if tuning_norm is None or len(tuning_norm) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=23, color='gray')
        ax.set_title(title, fontsize=25)
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

    ax.set_title(title, fontsize=25, fontweight='bold')
    ax.set_xlim(pos_min, pos_max)

    # Y-axis: optionally show cell count
    if show_ncells:
        ax.set_yticks([0, n_cells])
        ax.set_yticklabels(['0', str(n_cells)], fontsize=18)
        if show_ncells_ylabel:
            ax.set_ylabel('# cells', fontsize=18, labelpad=2)
    else:
        ax.set_yticks([])

    # X-axis: optionally show position ticks
    if show_xaxis:
        ticks = np.array([30, 60, 90, 120])
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{t:.0f}' for t in ticks], fontsize=18)
    else:
        ax.set_xticks([])


def _draw_combined_figure(layer_data, bc, col_labels, title, fname,
                          save_dir, skip_existing, ncells_all_cols=False):
    """
    Combined figure: rows = L2/3 | L4 | L5 | L6 | All layers (pooled strip).
    Columns = panels (days or trial-bin windows).

    layer_data     : {layer_name: [panel_or_None, ...]}
      panel        : {'tuning_norm': ndarray, 'preferred_positions': ndarray}
    col_labels     : list of column header strings (len == n_cols)
    ncells_all_cols: if True, show n-cells tick on every column (use for trial-bin
                     figures where each bin may contain different animals/cell counts)
    """
    out = os.path.join(save_dir, fname)
    if skip_existing and os.path.exists(out):
        print(f'  Skipped (exists): {fname}')
        return

    n_cols = len(col_labels)
    if n_cols == 0:
        print(f'  Skipped (no data): {fname}')
        return
    n_layer_rows = len(LAYER_ORDER)
    n_rows = n_layer_rows + 1          # +1 bottom strip row

    # When every column carries y-tick numbers, give a little extra breathing room;
    # but suppress the '# cells' label text on inner columns to avoid gap crowding.
    wspace = 0.15 if ncells_all_cols else 0.05
    # Reserve left margin: enough for layer label + '# cells' ylabel of col 0.
    left_margin = 0.13

    fig = plt.figure(figsize=(3.5 * n_cols, 4.3 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.15, wspace=wspace,
                           left=left_margin, right=0.97, top=0.93, bottom=0.07)

    row_axes = {}   # row_axes[row][col] = ax — used later for row-label placement

    # ---- layer rows ----
    for row, layer in enumerate(LAYER_ORDER):
        row_axes[row] = {}
        for col in range(n_cols):
            ax = fig.add_subplot(gs[row, col])
            row_axes[row][col] = ax
            panels = layer_data.get(layer, [])
            panel  = panels[col] if col < len(panels) else None
            show_n = (col == 0) or ncells_all_cols
            if panel is not None:
                _draw_heatmap(ax, panel['tuning_norm'], panel['preferred_positions'],
                              bc,
                              show_pref_strip=(col == n_cols - 1),
                              show_ncells=show_n,
                              show_ncells_ylabel=(col == 0),
                              show_xaxis=False)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=23, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
            # Column title AFTER _draw_heatmap so it isn't overwritten
            if row == 0:
                ax.set_title(col_labels[col], fontsize=25, fontweight='bold')

    # ---- all-layers pooled strip (bottom row) ----
    row_axes[n_layer_rows] = {}
    for col in range(n_cols):
        ax = fig.add_subplot(gs[n_layer_rows, col])
        row_axes[n_layer_rows][col] = ax
        all_t, all_p = [], []
        for layer in LAYER_ORDER:
            panels = layer_data.get(layer, [])
            panel  = panels[col] if col < len(panels) else None
            if panel is not None:
                all_t.append(panel['tuning_norm'])
                all_p.append(panel['preferred_positions'])
        show_n = (col == 0) or ncells_all_cols
        if all_t:
            _draw_heatmap(ax,
                          np.concatenate(all_t, axis=0),
                          np.concatenate(all_p, axis=0),
                          bc,
                          show_pref_strip=(col == n_cols - 1),
                          show_ncells=show_n,
                          show_ncells_ylabel=(col == 0),
                          show_xaxis=True)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=23, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])

    # ---- row labels in the left margin (figure coordinates, no collision risk) ----
    row_label_names = list(LAYER_ORDER) + ['All layers']
    for row, label in enumerate(row_label_names):
        color = LAYER_COLORS.get(label, 'black')
        ax0 = row_axes[row][0]
        pos = ax0.get_position()          # axes bounding box in figure coords
        y_center = (pos.y0 + pos.y1) / 2
        # x=0.01 places label at far-left margin, well left of the '# cells' ylabel
        fig.text(0.01, y_center, label,
                 fontsize=23, fontweight='bold', color=color,
                 ha='left', va='center', transform=fig.transFigure)

    fig.supxlabel('VR position (cm)', fontsize=23, fontweight='bold', y=0.01)
    fig.suptitle(title, fontsize=25, fontweight='bold')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  Saved: {fname}')
    plt.close(fig)


def _days_to_layer_data(animal_days):
    """Convert {day_num: session_data} → layer_data, bc, col_labels."""
    days = sorted(animal_days.keys())
    bc = None
    layer_data = {layer: [] for layer in LAYER_ORDER}
    for day in days:
        sd = animal_days[day]
        if bc is None:
            bc = sd['bin_centers']
        for layer in LAYER_ORDER:
            if layer in sd['layers']:
                layer_data[layer].append({
                    'tuning_norm': sd['layers'][layer]['tuning_norm'],
                    'preferred_positions': sd['layers'][layer]['preferred_positions'],
                })
            else:
                layer_data[layer].append(None)
    return layer_data, bc, [f'Day {d}' for d in days]


def _across_animals_days_to_layer_data(by_animal):
    """Pool all animals per day → layer_data, bc, col_labels (with animal counts)."""
    days_present = sorted({d for ad in by_animal.values() for d in ad})
    bc = None
    layer_data = {layer: [] for layer in LAYER_ORDER}
    col_labels = []
    for day in days_present:
        sessions = [ad[day] for ad in by_animal.values() if day in ad]
        col_labels.append(f'Day {day}\n({len(sessions)} mice)')
        for sd in sessions:
            if bc is None:
                bc = sd['bin_centers']
        for layer in LAYER_ORDER:
            tunings, pps = [], []
            for sd in sessions:
                if layer in sd['layers']:
                    tunings.append(sd['layers'][layer]['tuning_norm'])
                    pps.append(sd['layers'][layer]['preferred_positions'])
            layer_data[layer].append(
                {'tuning_norm': np.concatenate(tunings, axis=0),
                 'preferred_positions': np.concatenate(pps, axis=0)}
                if tunings else None
            )
    return layer_data, bc, col_labels


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

    fig.supxlabel('VR position (cm)', fontsize=23, fontweight='bold')
    fig.suptitle(f'{animal} — {day}: Spatial tuning (reliable cells, sorted by pref. pos.)',
                 fontsize=25, fontweight='bold')
    if out:
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved: {fname}')
    plt.close(fig)


# ============================================================================
# FIGURE B: PER-ANIMAL COMBINED (rows=layers+strip, cols=days)
# ============================================================================

def plot_animal_response_combined(animal_days, animal_id, save_dir, skip_existing=True):
    """Rows = layers + all-layers strip. Columns = recording days."""
    layer_data, bc, col_labels = _days_to_layer_data(animal_days)
    _draw_combined_figure(
        layer_data, bc, col_labels,
        title=f'{animal_id}: spatial tuning across days',
        fname=f'{animal_id}_response_days.png',
        save_dir=save_dir, skip_existing=skip_existing,
    )


# ============================================================================
# FIGURE C: ACROSS-ANIMALS COMBINED (rows=layers+strip, cols=days)
# ============================================================================

def plot_across_animals_response_combined(by_animal, save_dir, skip_existing=True):
    """Rows = layers + all-layers strip. Columns = days (all animals pooled)."""
    layer_data, bc, col_labels = _across_animals_days_to_layer_data(by_animal)
    _draw_combined_figure(
        layer_data, bc, col_labels,
        title='Across animals: spatial tuning across days (pooled)',
        fname='across_animals_response_days.png',
        save_dir=save_dir, skip_existing=skip_existing,
    )






# ============================================================================
# TRIAL-BIN DATA LOADING
# ============================================================================

def load_session_data_by_trialbin(session_dir, bin_size=TRIAL_BIN_SIZE):
    """
    Like load_session_data but returns per-trial-bin tuning curves.
    norm_spatial_activity (N_cells, N_laps, N_bins) is sliced into
    consecutive non-overlapping windows of `bin_size` laps and averaged.

    Returns same structure as load_session_data, except each layer dict has:
      'trial_bins': list of dicts, each with {tuning_norm, preferred_positions, label}
      (preferred_positions and smi are shared across bins — from the full-session SMI)
    """
    smi_files = glob.glob(os.path.join(session_dir, '*_smi_results.h5'))
    preproc_files = glob.glob(os.path.join(session_dir, '*preproc*.h5'))
    if not smi_files or not preproc_files:
        raise FileNotFoundError('Missing h5 files')

    animal, day_str, day_num = _parse_animal_day_from_path(session_dir)

    with h5py.File(preproc_files[0], 'r') as f:
        norm_act = f['norm_spatial_activity'][:]   # (N_cells, N_laps, N_bins)
        bin_centers = f['bin_centers'][:]
        combined_reliable = f['combined_reliable'][:].astype(bool) if 'combined_reliable' in f else None

    _, n_laps, _ = norm_act.shape
    starts = list(range(0, n_laps, bin_size))
    bin_labels = [f'Laps {s+1}–{min(s+bin_size, n_laps)}' for s in starts]

    # Mean tuning per trial-bin: list of (N_cells, N_bins)
    binned_tunings = [
        np.nanmean(norm_act[:, s:s + bin_size, :], axis=1)
        for s in starts
    ]
    # Session average: mean over ALL laps
    session_avg_tuning_all = np.nanmean(norm_act, axis=1)   # (N_cells, N_bins)

    with h5py.File(smi_files[0], 'r') as f:
        layers = {}
        for lk in f['layer_smi'].keys():
            layer_name = lk.replace('_', '/')
            lg = f['layer_smi'][lk]

            if USE_COMBINED_RELIABLE:
                if combined_reliable is None:
                    continue
                layer_cell_indices = lg['cell_indices'][:].astype(int)
                rv = layer_cell_indices[combined_reliable[layer_cell_indices]]
            else:
                if 'reliable_valid_cells' not in lg:
                    continue
                rv = lg['reliable_valid_cells'][:].astype(int)

            if len(rv) == 0:
                continue

            # Preferred positions: recompute from session-average peak when using
            # combined_reliable (stored pp is indexed to reliable_valid_cells, not rv)
            if USE_COMBINED_RELIABLE:
                sa_all = session_avg_tuning_all[rv, :]
                sa_max_all = sa_all.max(axis=1, keepdims=True)
                sa_max_all[sa_max_all == 0] = 1
                sa_norm = sa_all / sa_max_all
                pp = bin_centers[np.argmax(sa_norm, axis=1)]
            else:
                pp = lg['preferred_positions'][:]

            trial_bins = []
            for label, bt in zip(bin_labels, binned_tunings):
                tuning = bt[rv, :]
                row_max = tuning.max(axis=1, keepdims=True)
                row_max[row_max == 0] = 1
                trial_bins.append({
                    'tuning_norm': tuning / row_max,
                    'preferred_positions': pp,
                    'label': label,
                })

            # Session-average panel for this layer
            sa = session_avg_tuning_all[rv, :]
            sa_max = sa.max(axis=1, keepdims=True)
            sa_max[sa_max == 0] = 1
            session_avg = {
                'tuning_norm': sa / sa_max,
                'preferred_positions': pp,
            }

            layers[layer_name] = {'trial_bins': trial_bins, 'session_avg': session_avg}

    return {
        'animal': animal, 'day': day_str, 'day_num': day_num,
        'session_dir': session_dir,
        'bin_centers': bin_centers, 'layers': layers,
        'bin_labels': bin_labels,
    }


def load_all_sessions_trialbins(bin_size=TRIAL_BIN_SIZE):
    """
    Load trial-bin data for every session in SESSION_DIRS.
    Returns:
      by_session_tb : list of trial-bin session dicts
      by_animal_tb  : {animal: [session_tb, ...]}
    """
    by_session_tb = []
    by_animal_tb = {}
    for session_dir in SESSION_DIRS:
        if not os.path.isdir(session_dir):
            continue
        if not glob.glob(os.path.join(session_dir, '*_smi_results.h5')):
            continue
        try:
            sd = load_session_data_by_trialbin(session_dir, bin_size)
            by_session_tb.append(sd)
            by_animal_tb.setdefault(sd['animal'], []).append(sd)
            print(f'  [trial-bins] Loaded {sd["animal"]} {sd["day"]}')
        except Exception as e:
            print(f'  [trial-bins] ERROR {os.path.basename(session_dir)}: {e}')
    return by_session_tb, by_animal_tb


def _pool_trialbins_maxlaps(session_list, bin_size=TRIAL_BIN_SIZE):
    """
    Pool trial-bin data using the MAX number of laps across all sessions.
    Later bins will have fewer contributing sessions; the column label shows
    unique animals AND how many sessions each animal contributes to that bin
    (e.g. '044×2' means 2 recording days of JSY044 are included).
    Returns layer_data, bc, col_labels.
    """
    if not session_list:
        return {}, None, []

    n_bins_max = max(len(sd['bin_labels']) for sd in session_list)
    bc = session_list[0]['bin_centers']

    # Track sessions (animal+day tuples) rather than just animal names,
    # so we correctly reflect how many days of each animal contribute per bin.
    raw = {layer: [{'tuning': [], 'pp': [], 'sessions': set()}
                   for _ in range(n_bins_max)]
           for layer in LAYER_ORDER}

    for sd in session_list:
        n_this = len(sd['bin_labels'])
        session_key = (sd['animal'], sd['day'])
        for layer in LAYER_ORDER:
            if layer not in sd['layers']:
                continue
            tbs = sd['layers'][layer]['trial_bins']
            for b in range(n_this):
                raw[layer][b]['tuning'].append(tbs[b]['tuning_norm'])
                raw[layer][b]['pp'].append(tbs[b]['preferred_positions'])
                raw[layer][b]['sessions'].add(session_key)

    layer_data = {}
    for layer in LAYER_ORDER:
        bins_out = []
        for b in range(n_bins_max):
            entry = raw[layer][b]
            if entry['tuning']:
                bins_out.append({
                    'tuning_norm': np.concatenate(entry['tuning'], axis=0),
                    'preferred_positions': np.concatenate(entry['pp'], axis=0),
                })
            else:
                bins_out.append(None)
        layer_data[layer] = bins_out

    # Per-bin label: union of sessions across all layers, grouped by animal.
    # Format: '044×2, 052×1' to show how many days each animal contributes.
    col_labels = []
    for b in range(n_bins_max):
        sessions_b = set()
        for layer in LAYER_ORDER:
            sessions_b |= raw[layer][b]['sessions']
        # Count days per animal
        days_per_animal = Counter(animal for animal, _ in sessions_b)
        parts = []
        for animal in sorted(days_per_animal):
            num = ''.join(c for c in animal if c.isdigit())
            cnt = days_per_animal[animal]
            parts.append(f'{num}×{cnt}' if cnt > 1 else num)
        ids = ', '.join(parts)
        col_labels.append(
            f'Laps {b*bin_size+1}–{(b+1)*bin_size}\n'
            f'n={len(sessions_b)}: {ids}'
        )
    return layer_data, bc, col_labels


def _session_to_layer_data_trialbins(sd):
    """Convert a single session-tb dict to layer_data + col_labels.
    The final column is the session average (all laps)."""
    n_bins = len(sd['bin_labels'])
    layer_data = {}
    for layer in LAYER_ORDER:
        if layer not in sd['layers']:
            layer_data[layer] = [None] * (n_bins + 1)
            continue
        bins = [
            {'tuning_norm': tb['tuning_norm'],
             'preferred_positions': tb['preferred_positions']}
            for tb in sd['layers'][layer]['trial_bins']
        ]
        bins.append(sd['layers'][layer]['session_avg'])   # rightmost column
        layer_data[layer] = bins
    col_labels = list(sd['bin_labels']) + ['Session\navg']
    return layer_data, sd['bin_centers'], col_labels


# ============================================================================
# FIGURE G: PER-SESSION TRIAL-BIN COMBINED (rows=layers+strip, cols=trial bins)
# ============================================================================

def plot_session_trialbins_combined(session_dir, save_dir,
                                    bin_size=TRIAL_BIN_SIZE, skip_existing=True):
    """Combined grid+strip for one session. Saved as {animal}_{day}_response_trials.png"""
    try:
        sd = load_session_data_by_trialbin(session_dir, bin_size)
    except Exception as e:
        print(f'  ERROR loading {os.path.basename(session_dir)}: {e}')
        return
    animal, day = sd['animal'], sd['day']
    layer_data, bc, col_labels = _session_to_layer_data_trialbins(sd)
    _draw_combined_figure(
        layer_data, bc, col_labels,
        title=f'{animal} {day}: spatial tuning across trials ({bin_size}-lap bins)',
        fname=f'{animal}_{day}_response_trials.png',
        save_dir=save_dir, skip_existing=skip_existing,
        ncells_all_cols=True,
    )


# ============================================================================
# FIGURE H: PER-ANIMAL TRIAL-BIN COMBINED (all days pooled, max laps)
# ============================================================================

def plot_animal_trialbins_combined(animal_id, session_list, save_dir,
                                   bin_size=TRIAL_BIN_SIZE, skip_existing=True):
    """Per-animal combined grid+strip pooling all days, max-laps. {animal}_response_trials.png"""
    layer_data, bc, col_labels = _pool_trialbins_maxlaps(session_list, bin_size)
    if not col_labels:
        return
    _draw_combined_figure(
        layer_data, bc, col_labels,
        title=f'{animal_id}: spatial tuning across trials (all days pooled)',
        fname=f'{animal_id}_response_trials.png',
        save_dir=save_dir, skip_existing=skip_existing,
        ncells_all_cols=True,
    )


# ============================================================================
# FIGURE I: ACROSS-ANIMALS TRIAL-BIN COMBINED (all animals+days, max laps)
# ============================================================================

def plot_across_animals_trialbins_combined(all_sessions, save_dir,
                                           bin_size=TRIAL_BIN_SIZE, skip_existing=True):
    """Across-animals combined grid+strip, max-laps with per-bin animal count."""
    layer_data, bc, col_labels = _pool_trialbins_maxlaps(all_sessions, bin_size)
    if not col_labels:
        return
    n_animals = len({sd['animal'] for sd in all_sessions})
    _draw_combined_figure(
        layer_data, bc, col_labels,
        title=f'All animals ({n_animals} mice): spatial tuning across trials (all days pooled)',
        fname='across_animals_response_trials.png',
        save_dir=save_dir, skip_existing=skip_existing,
        ncells_all_cols=True,
    )


# ============================================================================
# FIGURE J: PER-DAY ACROSS-ANIMALS TRIAL-BIN COMBINED
# ============================================================================

def plot_per_day_trialbins_across_animals(by_session_tb, save_dir,
                                          bin_size=TRIAL_BIN_SIZE, skip_existing=True):
    """
    For each recording day: one combined figure (rows=layers+strip, cols=trial bins).
    All animals recorded on that day are pooled using max laps — later bins show
    fewer animals as shorter recordings drop out. Column headers show n per bin.
    Saved as Day{N}_response_trials.png.
    """
    # Group sessions by day number
    by_day = {}
    for sd in by_session_tb:
        by_day.setdefault(sd['day_num'], []).append(sd)

    for day_num, sessions in sorted(by_day.items()):
        layer_data, bc, col_labels = _pool_trialbins_maxlaps(sessions, bin_size)
        if not col_labels:
            continue
        n_animals = len({sd['animal'] for sd in sessions})

        # Append pooled session average across all animals on this day as final column
        for layer in LAYER_ORDER:
            tunings, pps = [], []
            for sd in sessions:
                if layer in sd['layers'] and 'session_avg' in sd['layers'][layer]:
                    tunings.append(sd['layers'][layer]['session_avg']['tuning_norm'])
                    pps.append(sd['layers'][layer]['session_avg']['preferred_positions'])
            layer_data[layer].append(
                {'tuning_norm': np.concatenate(tunings, axis=0),
                 'preferred_positions': np.concatenate(pps, axis=0)}
                if tunings else None
            )
        col_labels.append(f'Session\navg\n({n_animals} mice)')

        _draw_combined_figure(
            layer_data, bc, col_labels,
            title=f'Day {day_num}: spatial tuning across trials '
                  f'({n_animals} animals, {bin_size}-lap bins)',
            fname=f'Day{day_num}_response_trials.png',
            save_dir=save_dir, skip_existing=skip_existing,
            ncells_all_cols=True,
        )


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_all(skip_existing=False):
    summary_dir = r'D:\V1_SpatialModulation\2p\V1_prism\response_plots_wtO4144_combined_reliable'
    os.makedirs(summary_dir, exist_ok=True)

    print('\n' + '=' * 80)
    print(' SPATIAL RESPONSE PLOTS — ALL SESSIONS & SUMMARY FIGURES')
    print('=' * 80)

    print('\nLoading all sessions...')
    by_session, by_animal = load_all_sessions()
    print(f'  Loaded {len(by_session)} sessions from {len(by_animal)} animals\n')

    # --- Figure B: per-animal combined days (layers+strip × days) ---
    print('\n--- Figure B: Per-animal combined response (layers+strip × days) ---')
    for animal, animal_days in by_animal.items():
        animal_save = ANIMAL_DIRS.get(animal, summary_dir)
        os.makedirs(animal_save, exist_ok=True)
        plot_animal_response_combined(animal_days, animal, animal_save,
                                      skip_existing=skip_existing)
        plot_animal_response_combined(animal_days, animal, summary_dir,
                                      skip_existing=skip_existing)

    # --- Figure C: across-animals combined days ---
    print('\n--- Figure C: Across-animals combined response (layers+strip × days) ---')
    plot_across_animals_response_combined(by_animal, summary_dir,
                                          skip_existing=skip_existing)

    # --- Figures G/H/I: trial-bin combined figures ---
    print('\n--- Trial-bin figures: loading all sessions... ---')
    by_session_tb, by_animal_tb = load_all_sessions_trialbins()

    # print('\n--- Figure G: Per-session trial-bin combined ---')
    # for sd in by_session_tb:
    #     plot_session_trialbins_combined(sd['session_dir'], sd['session_dir'],
    #                                     skip_existing=skip_existing)
    #     plot_session_trialbins_combined(sd['session_dir'], summary_dir,
    #                                     skip_existing=skip_existing)

    print('\n--- Figure H: Per-animal trial-bin combined (all days pooled) ---')
    for animal, session_list in by_animal_tb.items():
        animal_save = ANIMAL_DIRS.get(animal, summary_dir)
        os.makedirs(animal_save, exist_ok=True)
        plot_animal_trialbins_combined(animal, session_list, animal_save,
                                       skip_existing=skip_existing)
        plot_animal_trialbins_combined(animal, session_list, summary_dir,
                                       skip_existing=skip_existing)

    print('\n--- Figure I: Across-animals trial-bin combined (max laps, all days pooled) ---')
    plot_across_animals_trialbins_combined(by_session_tb, summary_dir,
                                           skip_existing=skip_existing)

    print('\n--- Figure J: Per-day across-animals trial-bin combined (max laps) ---')
    plot_per_day_trialbins_across_animals(by_session_tb, summary_dir,
                                          skip_existing=skip_existing)

    print(f'\nSummary figures saved to: {summary_dir}')
    print('Done.')


def run_single_session(session_dir, skip_existing=False):
    """
    Plot trial-bin + session-average figure for a single session directory.
    Saves to the session folder and to the summary folder.
    Use this when testing on one recording before running run_all().
    """
    summary_dir = r'D:\V1_SpatialModulation\2p\V1_prism\response_plots'
    os.makedirs(summary_dir, exist_ok=True)

    active_sessions = [s for s in [session_dir] if os.path.isdir(s)]
    if not active_sessions:
        print(f'ERROR: session directory not found:\n  {session_dir}')
        return

    if not glob.glob(os.path.join(session_dir, '*_smi_results.h5')):
        print(f'ERROR: no *_smi_results.h5 in:\n  {session_dir}')
        print('Run preprocessing / SMI analysis first.')
        return

    print(f'\nLoading: {os.path.basename(session_dir)}')
    try:
        sd_tb = load_session_data_by_trialbin(session_dir)
    except Exception as e:
        print(f'ERROR loading session: {e}')
        traceback.print_exc()
        return

    print(f'  {sd_tb["animal"]} {sd_tb["day"]}  ({len(sd_tb["bin_labels"])} trial bins)')

    for save_dir in [session_dir, summary_dir]:
        plot_session_trialbins_combined(session_dir, save_dir,
                                        skip_existing=skip_existing)

    print('Done.')


if __name__ == '__main__':
    # --- Single-session mode -------------------------------------------
    # Uncomment and set a path to quickly plot one recording:
    # run_single_session(r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\...\TSeries-...')

    # --- Full run across all sessions in SESSION_DIRS ------------------
    run_all(skip_existing=False)
