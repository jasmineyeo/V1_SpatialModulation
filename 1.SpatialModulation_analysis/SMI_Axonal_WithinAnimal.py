"""
SMI_Axonal_WithinAnimal.py
Within-animal axonal analysis across days.

Generates three figures per animal:
  A. SMI trajectory (median ± MAD, Days 1–7)
  B. Peak position distribution heatmap (day × track position)
  C. Zone proportions per day (onset / mid-track / reward stacked bar)

Run with ANIMALS = ['JSY061'] or ['JSY060'] or both.
JSY, 03/2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
import h5py

from helper import files


# =============================================================================
# Animal configurations
# =============================================================================

BASE_61 = r'D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window'
BASE_60 = r'D:\V1_SpatialModulation\2p\V1_axonal\JSY060_ChronicImaging_prism'

ANIMAL_CONFIG = {
    'JSY061': {
        'imaging_type': 'L1_axonal',
        'label': 'JSY061 — L1 axonal (window)',
        'color': '#1565C0',
        'session_dirs': [
            rf'{BASE_61}\260202_JSY_JSY061_SpMod_AxonalImaging_Day1\TSeries-02022026-1804-001',
            rf'{BASE_61}\260203_JSY_JSY061_SpMod_AxonalImaging_Day2\TSeries-02032026-1751-001',
            rf'{BASE_61}\260204_JSY_JSY061_SpMod_AxonalImaging_Day3\TSeries-02042026-2009-001',
            rf'{BASE_61}\260205_JSY_JSY061_SpMod_AxonalImaging_Day4\TSeries-02052026-1833-002',
            rf'{BASE_61}\260206_JSY_JSY061_SpMod_AxonalImaging_Day5\TSeries-02062026-1850-001',
            rf'{BASE_61}\260207_JSY_JSY061_SpMod_AxonalImaging_Day6\TSeries-02072026-2023-001',
            rf'{BASE_61}\260208_JSY_JSY061_SpMod_AxonalImaging_Day7\TSeries-02082026-1826-001',
        ],
        'save_dir': rf'{BASE_61}\Axonal_Analysis',
    },
    'JSY060': {
        'imaging_type': 'L6_axonal',
        'label': 'JSY060 — L6 axonal (prism)',
        'color': '#B71C1C',
        'session_dirs': [
            rf'{BASE_60}\260225_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day1\TSeries-02252026-0903-001',
            rf'{BASE_60}\260226_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day2\TSeries-02262026-0915-001',
            rf'{BASE_60}\260227_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day3\TSeries-02262026-1253-001',
            rf'{BASE_60}\260228_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day4\TSeries-02282026-0919-001',
            rf'{BASE_60}\260301_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day5\TSeries-03012026-0914-002',
            rf'{BASE_60}\260302_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day6\TSeries-03022026-1226-001',
            rf'{BASE_60}\260303_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day7\TSeries-03032026-0817-001',
        ],
        'save_dir': rf'{BASE_60}\Axonal_Analysis',
    },
}


# =============================================================================
# Data loading
# =============================================================================

def _extract_day(session_dir):
    parent = os.path.basename(os.path.dirname(session_dir))
    match = re.search(r'Day(\d+)', parent)
    return int(match.group(1)) if match else None


def load_session_data(session_dir):
    """
    Load SMI h5 and preproc h5 for one session.
    Returns a dict with all data needed for plotting, or None if files missing.
    """
    smi_files = glob.glob(os.path.join(session_dir, '*_smi_results.h5'))
    preproc_files = glob.glob(os.path.join(session_dir, '*preproc*.h5'))

    if not smi_files:
        print(f"  No SMI h5 found in {os.path.basename(session_dir)}")
        return None
    if not preproc_files:
        print(f"  No preproc h5 found in {os.path.basename(session_dir)}")
        return None

    # --- Load SMI results ---
    smi = {}
    with h5py.File(smi_files[0], 'r') as f:
        smi['median'] = f['global_smi'].attrs['median_smi']
        smi['mad']    = f['global_smi'].attrs['mad_smi']
        smi['mean']   = f['global_smi'].attrs['mean_smi']
        smi['n_valid']= f['global_smi'].attrs['n_valid_cells']

        smi_vals = f['global_smi/SMI_all_cells'][:]
        valid_mask = f['global_smi/valid_cells_mask'][:]
        smi['smi_values'] = smi_vals[valid_mask]
        smi['smi_values'] = smi['smi_values'][
            ~np.isnan(smi['smi_values']) & ~np.isinf(smi['smi_values'])
        ]

        smi['n_analyzed'] = int(f.attrs['n_cells_analyzed'])

    # --- Load preproc for peak positions of all reliable cells ---
    preproc = files.read_h5(preproc_files[0])
    spatial_activity = preproc['spatial_activity']       # (n_cells, n_trials, n_bins)
    bin_centers      = preproc['bin_centers']            # (n_bins,) in cm
    reliable_cells   = preproc['combined_reliable'].astype(bool)

    # Mean across trials for peak finding
    mean_profile = np.mean(spatial_activity, axis=1)    # (n_cells, n_bins)
    peak_bins = np.argmax(mean_profile, axis=1)          # (n_cells,)
    peak_positions_all = bin_centers[peak_bins]          # cm, all cells

    # Only reliable cells
    peak_positions = peak_positions_all[reliable_cells]

    # Zone boundaries: onset < 20 cm, reward > 115 cm (last landmark), mid = rest
    onset_thresh_zone  = 20.0
    reward_thresh_zone = 115.0

    n_reliable = len(peak_positions)
    smi['peak_positions']   = peak_positions
    smi['bin_centers']      = bin_centers
    smi['onset_thresh']     = onset_thresh_zone
    smi['reward_thresh']    = reward_thresh_zone
    smi['n_reliable']       = n_reliable
    smi['n_onset']          = int(np.sum(peak_positions < onset_thresh_zone))
    smi['n_reward']         = int(np.sum(peak_positions > reward_thresh_zone))
    smi['n_mid']            = int(np.sum(
        (peak_positions >= onset_thresh_zone) & (peak_positions <= reward_thresh_zone)
    ))

    return smi


def load_animal_data(animal_id):
    config = ANIMAL_CONFIG[animal_id]
    sessions = []
    for session_dir in config['session_dirs']:
        day = _extract_day(session_dir)
        if day is None:
            continue
        print(f"  Loading Day {day}: {os.path.basename(session_dir)}")
        data = load_session_data(session_dir)
        if data is not None:
            data['day'] = day
            sessions.append(data)
    sessions.sort(key=lambda x: x['day'])
    return sessions


# =============================================================================
# Figure A: SMI trajectory
# =============================================================================

def plot_smi_trajectory(sessions_dict, save_path):
    """
    Line plot of median SMI ± MAD across days for each animal.
    sessions_dict: {animal_id: [session_data, ...]}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for animal_id, sessions in sessions_dict.items():
        cfg = ANIMAL_CONFIG[animal_id]
        days    = [s['day'] for s in sessions]
        medians = [s['median'] for s in sessions]
        mads    = [s['mad']    for s in sessions]

        ax.plot(days, medians, '-o', color=cfg['color'], label=cfg['label'], linewidth=2, markersize=7)
        ax.fill_between(
            days,
            [m - e for m, e in zip(medians, mads)],
            [m + e for m, e in zip(medians, mads)],
            color=cfg['color'], alpha=0.15
        )

    ax.set_xlabel('Day', fontsize=13)
    ax.set_ylabel('Median SMI ± MAD', fontsize=13)
    ax.set_title('SMI Trajectory — RSC Axonal Projections to V1', fontsize=14)
    ax.set_xticks(sorted({s['day'] for ss in sessions_dict.values() for s in ss}))
    ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    fpath = os.path.join(save_path, 'A_smi_trajectory.png')
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(fpath)}")
    return fig


# =============================================================================
# Figure B: Peak position distribution heatmap
# =============================================================================

def plot_peak_distribution_heatmap(sessions_dict, save_path):
    """
    2D heatmap: rows = days, cols = track position.
    Colour = proportion of reliable cells peaking at that position.
    One subplot per animal.
    """
    n_animals = len(sessions_dict)
    fig, axes = plt.subplots(1, n_animals, figsize=(7 * n_animals, 5), squeeze=False)

    # Position bins for histogram (5 cm bins, 0–135 cm)
    pos_edges = np.arange(0, 136, 5)
    pos_centers = (pos_edges[:-1] + pos_edges[1:]) / 2

    for col, (animal_id, sessions) in enumerate(sessions_dict.items()):
        cfg = ANIMAL_CONFIG[animal_id]
        ax  = axes[0, col]

        days = [s['day'] for s in sessions]
        n_days = len(days)

        heatmap = np.zeros((n_days, len(pos_centers)))
        for row, sess in enumerate(sessions):
            counts, _ = np.histogram(sess['peak_positions'], bins=pos_edges)
            n_total = len(sess['peak_positions'])
            heatmap[row] = counts / n_total if n_total > 0 else 0

        im = ax.imshow(
            heatmap, aspect='auto', origin='upper',
            extent=[pos_edges[0], pos_edges[-1], days[-1] + 0.5, days[0] - 0.5],
            cmap='YlOrRd', vmin=0, vmax=heatmap.max()
        )
        plt.colorbar(im, ax=ax, label='Proportion of cells')

        # Landmark lines
        for lm_pos in [25, 55, 85, 115]:
            ax.axvline(lm_pos, color='white', linestyle='--', alpha=0.6, linewidth=1)

        ax.set_xlabel('Track Position (cm)', fontsize=12)
        ax.set_ylabel('Day', fontsize=12)
        ax.set_yticks(days)
        ax.set_title(cfg['label'], fontsize=12)

    fig.suptitle('Peak Position Distribution Across Days', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fpath = os.path.join(save_path, 'B_peak_distribution_heatmap.png')
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(fpath)}")
    return fig


# =============================================================================
# Figure C: Zone proportions per day
# =============================================================================

def plot_zone_proportions(sessions_dict, save_path):
    """
    Stacked bar chart: proportion of onset / mid-track / reward cells per day.
    """
    n_animals = len(sessions_dict)
    fig, axes = plt.subplots(1, n_animals, figsize=(6 * n_animals, 5), squeeze=False)

    zone_colors = {
        'Onset (<20 cm)':     '#E53935',
        'Mid-track':          '#43A047',
        'Reward (>115 cm)':   '#1E88E5',
    }

    for col, (animal_id, sessions) in enumerate(sessions_dict.items()):
        cfg = ANIMAL_CONFIG[animal_id]
        ax  = axes[0, col]

        days = [s['day'] for s in sessions]
        x    = np.arange(len(days))
        width = 0.6

        p_onset  = np.array([s['n_onset']  / s['n_reliable'] if s['n_reliable'] > 0 else 0 for s in sessions])
        p_reward = np.array([s['n_reward'] / s['n_reliable'] if s['n_reliable'] > 0 else 0 for s in sessions])
        p_mid    = np.array([s['n_mid']    / s['n_reliable'] if s['n_reliable'] > 0 else 0 for s in sessions])

        bottom = np.zeros(len(days))
        for label, prop, color in [
            ('Onset (<20 cm)',   p_onset,  zone_colors['Onset (<20 cm)']),
            ('Mid-track',        p_mid,    zone_colors['Mid-track']),
            ('Reward (>115 cm)', p_reward, zone_colors['Reward (>115 cm)']),
        ]:
            ax.bar(x, prop, width, bottom=bottom, label=label, color=color, alpha=0.85)
            bottom += prop

        ax.set_xticks(x)
        ax.set_xticklabels([f'Day {d}' for d in days], rotation=30, ha='right')
        ax.set_ylabel('Proportion of reliable cells', fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title(cfg['label'], fontsize=12)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Cell Zone Distribution Across Days', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fpath = os.path.join(save_path, 'C_zone_proportions.png')
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(fpath)}")
    return fig


# =============================================================================
# Main
# =============================================================================

def run_axonal_within_animal(animal_ids=None):
    if animal_ids is None:
        animal_ids = list(ANIMAL_CONFIG.keys())

    # Determine shared save path (next to script, or first animal's save_dir)
    shared_save = os.path.join(
        os.path.dirname(__file__), 'Axonal_WithinAnimal_Figures'
    )
    os.makedirs(shared_save, exist_ok=True)

    sessions_dict = {}
    for animal_id in animal_ids:
        print(f"\n{'='*60}")
        print(f"Loading {animal_id} ({ANIMAL_CONFIG[animal_id]['imaging_type']})")
        print(f"{'='*60}")
        sessions = load_animal_data(animal_id)
        if not sessions:
            print(f"  No data found for {animal_id}, skipping.")
            continue
        sessions_dict[animal_id] = sessions

        # Also save per-animal figures
        per_animal_dir = ANIMAL_CONFIG[animal_id]['save_dir']
        os.makedirs(per_animal_dir, exist_ok=True)

    if not sessions_dict:
        print("No data loaded. Exiting.")
        return

    print(f"\n{'='*60}")
    print("Generating figures...")
    print(f"{'='*60}")

    plot_smi_trajectory(sessions_dict, shared_save)
    plot_peak_distribution_heatmap(sessions_dict, shared_save)
    plot_zone_proportions(sessions_dict, shared_save)

    print(f"\nAll figures saved to: {shared_save}")


if __name__ == "__main__":
    # Run both animals together for side-by-side comparison
    # Change to ['JSY061'] or ['JSY060'] to run a single animal
    run_axonal_within_animal(animal_ids=['JSY061', 'JSY060'])
