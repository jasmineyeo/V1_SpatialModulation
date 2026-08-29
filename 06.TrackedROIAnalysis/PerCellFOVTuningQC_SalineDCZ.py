"""
PerCellFOVTuningQC_SalineDCZ.py
For every cell TrackROIs_SalineDCZ.py tracked across SALINE and DCZ
(regardless of whether it later survives CompareTrackedLayerSMI_SalineDCZ.py's
layer-consistency/reliability filters), save one QC figure showing:
  - top row:    the cell's ROI footprint highlighted on the FULL mean FOV,
                one panel per session
  - bottom row: the cell's spatial tuning curve (activity vs. position —
                the curve SMI is computed from), one panel per session
  - title:      layer assignment + SMI (global and layer-reliable) per
                session, flagging layer mismatches and missing-reliability
                cells rather than silently dropping them (unlike
                CompareTrackedLayerSMI_SalineDCZ.py, this script is meant
                for auditing, so it covers every tracked cell).

Batch mode — loops over every tracked cell and saves a PNG per cell to
`PerCellFOVTuningQC/` next to roi_tracking_results.h5. No interactive
review; open the folder afterward to browse.

Pipeline this sits alongside:
  1. SMI_FullSession_Interactive.py -> smi_results.h5 per session
  2. TrackROIs_SalineDCZ.py         -> roi_tracking_results.h5
  3. MergeTrackedLayerSMI.py        -> tidy tracked_cell_row x day_label table
  4. THIS SCRIPT                    -> per-cell FOV + tuning-curve QC figures
     (CompareTrackedLayerSMI_SalineDCZ.py is the paired-stats sibling of this)

JSY, 2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import glob
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')  # batch save, no interactive window needed
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from helper import files
from MergeTrackedLayerSMI import merge_tracked_layer_smi, _decode


# ============================================================
# CONFIGURATION
# ============================================================
tracking_h5_path = r"D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260724_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_1\TrackedROIs\roi_tracking_results.h5"

required_days = ['SALINE', 'DCZ']


# ============================================================
# Function 1: Session plane0 paths (from the tracking h5)
# ============================================================
def get_session_plane0_paths(tracking_h5_path, required_days):
    with h5py.File(tracking_h5_path, 'r') as f:
        return {day: _decode(f[f'sessions/{day}'].attrs['plane0_path'])
                for day in required_days}


# ============================================================
# Function 2: Locate + load a session's preproc.h5
# ============================================================
def find_preproc_path(plane0_path):
    tseries_dir = os.path.dirname(os.path.dirname(str(plane0_path)))
    matches = glob.glob(os.path.join(tseries_dir, "*preproc*.h5"))
    return matches[0] if matches else None


# ============================================================
# Function 3: Load everything needed to plot cells for one session
# (mean FOV + per-ROI footprints from suite2p, spatial tuning curves
# from preproc.h5 — loaded once per session, reused for every cell)
# ============================================================
def load_session_bundle(plane0_path):
    plane0_path = str(plane0_path)

    stat = np.load(os.path.join(plane0_path, 'stat.npy'), allow_pickle=True)
    iscell = np.load(os.path.join(plane0_path, 'iscell.npy'), allow_pickle=True)
    ops = np.load(os.path.join(plane0_path, 'ops.npy'), allow_pickle=True).item()

    cell_idx = np.where(iscell[:, 0] == 1)[0]
    stat_cells = [stat[i] for i in cell_idx]

    preproc_path = find_preproc_path(plane0_path)
    if preproc_path is None:
        raise FileNotFoundError(f"No *preproc*.h5 found next to {plane0_path}")
    preproc = files.read_h5(preproc_path)

    return {
        'stat': stat_cells,
        'mean_img': ops['meanImg'],
        'spatial_activity': preproc['spatial_activity'],
        'bin_centers': preproc['bin_centers'],
    }


# ============================================================
# Function 4: Plot one cell — FOV + tuning curve, both sessions
# ============================================================
def plot_cell_fov_and_tuning(row_data, session_bundles, day_a, day_b, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for col, day in enumerate([day_a, day_b]):
        bundle = session_bundles[day]
        roi_idx = int(row_data[f'roi_idx_{day}'])
        s = bundle['stat'][roi_idx]
        ypix, xpix = s['ypix'], s['xpix']
        lam_norm = s['lam'] / s['lam'].max()

        # --- Top row: full FOV with this ROI highlighted ---
        ax_fov = axes[0, col]
        mean_img = bundle['mean_img']
        ax_fov.imshow(mean_img, cmap='gray',
                      vmin=np.percentile(mean_img, 2), vmax=np.percentile(mean_img, 98))

        overlay = np.zeros((*mean_img.shape, 4))
        overlay[ypix, xpix, 0] = 1.0
        overlay[ypix, xpix, 3] = np.clip(lam_norm * 0.9 + 0.3, 0, 1)
        ax_fov.imshow(overlay)

        # Locator ring so the ROI is visible even when tiny at full-FOV scale
        med_y, med_x = s['med']
        ax_fov.add_patch(Circle((med_x, med_y), radius=18, fill=False,
                                 edgecolor='yellow', linewidth=1.8))

        layer = row_data.get(f'layer_{day}', None)
        ax_fov.set_title(f"{day} — full FOV\nROI {roi_idx}, layer={layer}", fontsize=11)
        ax_fov.axis('off')

        # --- Bottom row: spatial tuning curve ---
        ax_tune = axes[1, col]
        tuning = bundle['spatial_activity'][roi_idx]  # (n_trials, n_bins)
        mean_tuning = np.nanmean(tuning, axis=0)
        bin_centers = bundle['bin_centers']

        ax_tune.plot(bin_centers, mean_tuning, color='steelblue', linewidth=2)

        pref_pos = row_data.get(f'preferred_position_{day}', None)
        if pref_pos is not None and not np.isnan(pref_pos):
            ax_tune.axvline(pref_pos, color='green', linestyle='--', alpha=0.8,
                             label=f'Preferred ({pref_pos:.1f} cm)')
            ax_tune.legend(fontsize=9, loc='upper right')

        smi_global = row_data.get(f'SMI_global_{day}', np.nan)
        smi_layer = row_data.get(f'SMI_layer_reliable_{day}', np.nan)
        rp = row_data.get(f'Rp_{day}', np.nan)
        rn = row_data.get(f'Rn_{day}', np.nan)

        smi_layer_str = f"{smi_layer:.3f}" if not np.isnan(smi_layer) else "not reliable"
        info = (f"SMI_global={smi_global:.3f}\n"
                f"SMI_layer_reliable={smi_layer_str}\n"
                f"Rp={rp:.3f}, Rn={rn:.3f}")
        ax_tune.text(0.02, 0.98, info, transform=ax_tune.transAxes,
                     fontsize=9, va='top', ha='left',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        ax_tune.set_xlabel('Position (cm)')
        ax_tune.set_ylabel('Activity')
        ax_tune.set_title(f"{day} — spatial tuning curve", fontsize=11)
        ax_tune.grid(True, alpha=0.3)

    layer_a = row_data.get(f'layer_{day_a}', None)
    layer_b = row_data.get(f'layer_{day_b}', None)
    mismatch_flag = "  *** LAYER MISMATCH ***" if layer_a != layer_b else ""

    fig.suptitle(f"Tracked cell row {row_data['tracked_cell_row']}   "
                 f"({day_a} layer={layer_a} / {day_b} layer={layer_b}){mismatch_flag}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    save_dir = os.path.join(os.path.dirname(tracking_h5_path), 'PerCellFOVTuningQC')
    os.makedirs(save_dir, exist_ok=True)

    day_a, day_b = required_days

    print("Loading tracked-cell x SMI table...")
    df = merge_tracked_layer_smi(tracking_h5_path, required_days=required_days)

    print("\nLoading session bundles (FOV + footprints + tuning curves)...")
    plane0_paths = get_session_plane0_paths(tracking_h5_path, required_days)
    session_bundles = {}
    for day in required_days:
        print(f"  {day}: {plane0_paths[day]}")
        session_bundles[day] = load_session_bundle(plane0_paths[day])

    wide = df.pivot(index='tracked_cell_row', columns='day_label')
    wide.columns = [f'{col}_{day}' for col, day in wide.columns]
    wide = wide.reset_index()

    n_cells = len(wide)
    print(f"\nSaving {n_cells} per-cell QC figures to {save_dir}...")

    for i, row in wide.iterrows():
        row_data = row.to_dict()
        save_path = os.path.join(save_dir, f"cell_{int(row_data['tracked_cell_row']):04d}.png")
        plot_cell_fov_and_tuning(row_data, session_bundles, day_a, day_b, save_path)

        if (i + 1) % 20 == 0 or (i + 1) == n_cells:
            print(f"  {i+1}/{n_cells} saved...")

    print(f"\nDone. {n_cells} figures saved to {save_dir}")
