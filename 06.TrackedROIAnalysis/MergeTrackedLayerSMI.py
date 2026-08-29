"""
MergeTrackedLayerSMI.py
Joins TrackROIs.py's tracked-cell registration matrix with each session's own
smi_results.h5 (layer-specific SMI results) into one tidy table:
tracked_cell_row x day_label x layer/SMI features.

This is a pure index-based join — TrackROIs.py already establishes cell
correspondence across sessions (registration_matrix / verified_matrix), and
each session's smi_results.h5 is ordered by the same iscell-filtered stat
array TrackROIs.py uses (see helper/twop.py find_files vs
TrackROIs.load_suite2p_session), so no coordinate transforms are needed here.

JSY, 2026
"""

import os
import glob
import numpy as np
import pandas as pd
import h5py


def _decode(x):
    """h5py attrs may come back as bytes depending on version — normalize to str."""
    return x.decode() if isinstance(x, bytes) else x


def _sanitize(name):
    return str(name).replace('/', '_').replace('\\', '_')


def find_smi_results_path(plane0_path):
    """Given a .../TSeries-.../suite2p/plane0 path, locate that session's smi_results.h5."""
    tseries_dir = os.path.dirname(os.path.dirname(str(plane0_path)))
    matches = glob.glob(os.path.join(tseries_dir, "*_smi_results.h5"))
    return matches[0] if matches else None


def load_smi_results(smi_path):
    """
    Load per-cell layer assignment + SMI features from one session's
    smi_results.h5, indexed by the same ROI index TrackROIs.py uses.
    """
    with h5py.File(smi_path, 'r') as f:
        global_grp = f['global_smi']
        SMI_all = global_grp['SMI_all_cells'][:]
        Rp = global_grp['Rp'][:]
        Rn = global_grp['Rn'][:]
        pref_pos = global_grp['preferred_positions'][:]

        layer_of_cell = {}       # roi_idx -> layer name (e.g. 'L4', or 'All' for window recordings)
        layer_reliable_smi = {}  # roi_idx -> SMI (only cells that were reliable+valid within their layer)

        layer_smi_grp = f['layer_smi']
        for safe_name in layer_smi_grp.keys():
            grp = layer_smi_grp[safe_name]
            orig_name = _decode(grp.attrs.get('original_name', safe_name))
            for idx in grp['cell_indices'][:]:
                layer_of_cell[int(idx)] = orig_name

            if 'reliable_valid_cells' in grp and 'SMI' in grp:
                for idx, smi in zip(grp['reliable_valid_cells'][:], grp['SMI'][:]):
                    layer_reliable_smi[int(idx)] = float(smi)

    return {
        'SMI_all': SMI_all, 'Rp': Rp, 'Rn': Rn, 'pref_pos': pref_pos,
        'layer_of_cell': layer_of_cell, 'layer_reliable_smi': layer_reliable_smi,
    }


def merge_tracked_layer_smi(tracking_h5_path, required_days=None):
    """
    Build a tidy DataFrame joining a TrackROIs.py tracking result with each
    tracked session's layer-specific SMI results.

    Parameters:
    -----------
    tracking_h5_path : str
        Path to the roi_tracking_results.h5 saved by TrackROIs.py.
    required_days : list of str, optional
        Which sessions a cell must be tracked across to be included. Defaults
        to the required_days TrackROIs.py was run with.

    Returns:
    --------
    df : pandas.DataFrame
        Columns: tracked_cell_row, day_label, roi_idx, layer, SMI_global,
        SMI_layer_reliable (NaN if the cell wasn't reliable+valid that
        session), Rp, Rn, preferred_position.
    """
    with h5py.File(tracking_h5_path, 'r') as f:
        verified_matrix = f['verified_matrix'][:]
        day_labels = [_decode(d) for d in f.attrs['day_labels']]
        stored_required_days = [_decode(d) for d in f.attrs['required_days']]

        session_plane0 = {day: _decode(f[f'sessions/{day}'].attrs['plane0_path'])
                           for day in day_labels}

    if required_days is None:
        required_days = stored_required_days
    required_cols = [day_labels.index(d) for d in required_days]

    mask = np.all(verified_matrix[:, required_cols] >= 0, axis=1)
    tracked_rows = np.where(mask)[0]
    print(f"Merging {len(tracked_rows)} cells tracked across {required_days}...")

    smi_cache = {}
    for day in required_days:
        smi_path = find_smi_results_path(session_plane0[day])
        if smi_path is None:
            raise FileNotFoundError(f"No *_smi_results.h5 found for session '{day}' "
                                     f"(looked next to {session_plane0[day]})")
        print(f"  {day}: {os.path.basename(smi_path)}")
        smi_cache[day] = load_smi_results(smi_path)

    records = []
    for row in tracked_rows:
        for day in required_days:
            col = day_labels.index(day)
            roi_idx = int(verified_matrix[row, col])
            smi = smi_cache[day]

            records.append({
                'tracked_cell_row': int(row),
                'day_label': day,
                'roi_idx': roi_idx,
                'layer': smi['layer_of_cell'].get(roi_idx, None),
                'SMI_global': float(smi['SMI_all'][roi_idx]),
                'SMI_layer_reliable': smi['layer_reliable_smi'].get(roi_idx, np.nan),
                'Rp': float(smi['Rp'][roi_idx]),
                'Rn': float(smi['Rn'][roi_idx]),
                'preferred_position': float(smi['pref_pos'][roi_idx]),
            })

    df = pd.DataFrame.from_records(records)

    n_layer_changes = (
        df.groupby('tracked_cell_row')['layer'].nunique().gt(1).sum()
    )
    print(f"\n{len(tracked_rows)} tracked cells merged across {len(required_days)} sessions.")
    print(f"  {n_layer_changes} cells were assigned to more than one layer across sessions "
          f"(check these against the boundary consistency QC plot before trusting them).")

    return df


if __name__ == '__main__':
    # --- Example usage ---
    # tracking_h5_path = r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\...\TrackedROIs\roi_tracking_results.h5'
    # df = merge_tracked_layer_smi(tracking_h5_path)
    # out_csv = os.path.join(os.path.dirname(tracking_h5_path), 'tracked_layer_smi.csv')
    # df.to_csv(out_csv, index=False)
    # print(f"Saved merged table to {out_csv}")
    pass
