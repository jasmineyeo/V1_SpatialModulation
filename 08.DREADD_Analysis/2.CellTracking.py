"""
2.CellTracking.py
Phase 2 (DREADD saline/DCZ cohort) -- Same-Cell Tracking Across Sessions.

Tracks individual cells across a caller-chosen SMALL group of sessions
(e.g. BaselineDay5 + DCZ1_SALINE + DCZ1_DCZ, or an openloop active
SALINE/DCZ pair) rather than across the full ~15-session series per
animal -- tracking everything at once collapses N down to whatever
survives every single session, which isn't realistic given the length of
this series. Each comparison group gets its own registration + master
cell-ID table, built on demand via `track_and_build_table` /
`TRACKING_GROUPS` below.

Built on top of the existing TrackROIs.py/TrackROIs_SalineDCZ.py machinery
(FOV alignment via phase_cross_correlation, centroid+footprint candidate
matching, Hungarian assignment, auto-accept/reject + manual review) --
reimplemented and generalized here to an arbitrary session list + caller-
chosen reference (rather than a hardcoded hand of 2 required sessions),
and joined against Phase 1's per-session layer labels
(1.LayerAssignment_Curve.py's *_layer_curve_results(_averaged).h5) to
produce the master table.

Output compatibility / design notes:
- Every roi_idx here is a POSITION in the iscell-filtered cell list
  (np.where(iscell[:,0]==1)[0]), built the same way by both this module's
  load_session_for_tracking and Phase 1's per-session loaders -- position i
  refers to the same physical cell in both, which is the join key
  build_master_cell_table relies on. A real bug was caught and fixed here:
  _load_layer_of_cell used to key its lookup by the raw suite2p stat.npy
  index instead of by position, silently producing wrong layer labels for
  most cells. Keep this convention in mind if this module is ever extended.
- Phase 1's boundary-curve consistency issue (a raw ~66% layer-mismatch
  rate on the first real tracked group) was NOT a Phase 2 problem at its
  root -- it traced to independently-clicked per-session curves drifting
  relative to each other. The fix lives in Phase 1
  (1.LayerAssignment_Curve.py's chained registration + averaged reference +
  vertical-nudge adjustment), not here. After that fix (plus the roi_idx
  bug fix above), a real tracked group came out at 3/119 mismatched cells
  -- consistent with genuine borderline cases, which flag_layer_mismatches/
  review_layer_mismatches_popup (Functions 2.8/2.9) exist to resolve.
- Track A (this module) is scoped per scientific comparison, not per
  animal -- see the TRACKING_GROUPS template in __main__ for the actual
  recommended set of comparisons. Track B (all-cells, session-independent
  SMI/landmark computation) needs none of this module at all; it runs
  directly off each session's own preproc.h5 in Phase 3.

Developed incrementally in 2.CellTracking.ipynb (same folder) -- that
notebook is the build record; this file is the consolidated, reusable
version for real use.

JSY, 2026
"""

import os
import re
import glob
import pickle
import json
import numpy as np
import h5py
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # interactive review popups
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.widgets import CheckButtons, Button
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation

rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20

MICRONS_PER_PIXEL = 1.08952017715202  # V1_prism_DREADD animals (JSY090, JSY093)


# ============================================================
# Function 2.1: session discovery (convenience, not the main driver)
# ============================================================
def discover_animal_sessions(animal_dir):
    """
    Scan animal_dir for every TSeries-*/suite2p/plane0 folder anywhere in
    its tree, labeling each by whichever known naming pattern it matches:

    - Folder name contains 'skip' (case-insensitive, in the TSeries or
      parent folder name) -> excluded entirely, not added to the catalog.
      For a Day/condition folder with 2 recordings where you only want one
      (e.g. a "no water delivered" recording alongside the real one) --
      rename the one to exclude to include "skip" and it's dropped here
      before label matching even happens.
    - TSeries folder name contains 'SAL' (covers SALINE/SAL) -> 'saline',
      label '{parent_folder}_SALINE'
    - TSeries folder name contains 'DCZ' -> 'dcz', label '{parent_folder}_DCZ'
    - parent folder name matches Day<N> (case-insensitive) and neither of
      the above matched -> 'baseline', label 'Day<N>'
    - anything else -> 'unknown', label = the raw TSeries folder name
      (printed distinctly so nothing gets silently mislabeled -- this is
      where openloop sessions land until a real pattern is added for them)

    Guards against label collisions as a safety net (e.g. two non-"skip"
    recordings that happen to compute the same label) -- the second one
    found is skipped with a warning rather than silently overwriting the
    first.

    Parameters
    ----------
    animal_dir : str
        Animal's top-level folder.

    Returns
    -------
    catalog : dict
        {label: {'plane0_path': str, 'session_type': str, 'tseries_dir': str}}
    """
    catalog = {}
    unmatched = []
    skipped = []

    plane0_paths = sorted(glob.glob(os.path.join(animal_dir, '**', 'suite2p', 'plane0'),
                                     recursive=True))

    for plane0_path in plane0_paths:
        tseries_dir = os.path.dirname(os.path.dirname(plane0_path))
        tseries_name = os.path.basename(tseries_dir)
        parent_dir = os.path.dirname(tseries_dir)
        parent_name = os.path.basename(parent_dir)

        if 'skip' in tseries_name.lower() or 'skip' in parent_name.lower():
            skipped.append(tseries_dir)
            continue

        upper_tseries = tseries_name.upper()

        if 'SAL' in upper_tseries:
            session_type = 'saline'
            label = f'{parent_name}_SALINE'
        elif 'DCZ' in upper_tseries:
            session_type = 'dcz'
            label = f'{parent_name}_DCZ'
        else:
            day_match = re.search(r'Day(\d+)', parent_name, re.IGNORECASE)
            if day_match:
                session_type = 'baseline'
                label = f'Day{day_match.group(1)}'
            else:
                session_type = 'unknown'
                label = tseries_name

        if label in catalog:
            print(f"WARNING: duplicate label '{label}' -- keeping "
                  f"{catalog[label]['tseries_dir']}, skipping {tseries_dir}")
            continue

        catalog[label] = {
            'plane0_path': plane0_path,
            'session_type': session_type,
            'tseries_dir': tseries_dir,
        }
        if session_type == 'unknown':
            unmatched.append(label)

    print(f"Discovered {len(catalog)} sessions under {animal_dir}:")
    for label, info in catalog.items():
        print(f"  [{info['session_type']:>8}] {label}  <-  {info['tseries_dir']}")

    if skipped:
        print(f"\n{len(skipped)} session(s) excluded via 'skip' in their folder name:")
        for s in skipped:
            print(f"  {s}")

    if unmatched:
        print(f"\n{len(unmatched)} session(s) didn't match a known naming pattern "
              f"(labeled 'unknown', tagged by raw TSeries folder name): {unmatched}")
        print("If any of these are openloop sessions, tell me their real folder "
              "naming convention and I'll add a pattern for them.")

    return catalog


# ============================================================
# Interactive session/reference picker (extends the Function 2.1/2.10
# workflow) -- placed here, right after discover_animal_sessions, since
# it's a companion to it: an alternative to hand-typing session_labels/
# reference_label into a TRACKING_GROUPS entry or SELECTED_LABELS variable.
# ============================================================
def select_sessions_popup(session_catalog, title='Select sessions for this tracking group'):
    """
    Checkbox popup listing every session in session_catalog. Click boxes to
    toggle, click 'Confirm selection' or press Enter when done. Closing
    without confirming just uses whatever was checked at that point, rather
    than erroring.

    Parameters
    ----------
    session_catalog : dict
        From discover_animal_sessions.
    title : str

    Returns
    -------
    selected_labels : list of str
        Labels you checked, in session_catalog's original order.
    """
    labels = list(session_catalog.keys())
    n = len(labels)
    display_labels = [f"[{session_catalog[l]['session_type']:>8}] {l}" for l in labels]

    fig_height = max(4, 0.35 * n + 1.5)
    fig = plt.figure(figsize=(9, fig_height))
    try:
        fig.canvas.manager.set_window_title('SELECT SESSIONS -- check boxes, then Confirm (or Enter)')
    except Exception:
        pass

    fig.suptitle(title, fontsize=12, fontweight='bold')

    check_ax = fig.add_axes([0.05, 0.12, 0.9, 0.80])
    check = CheckButtons(check_ax, display_labels, [False] * n)

    confirm_ax = fig.add_axes([0.35, 0.02, 0.3, 0.06])
    confirm_button = Button(confirm_ax, 'Confirm selection')

    state = {'done': False, 'status': [False] * n}

    def on_confirm(event=None):
        state['status'] = list(check.get_status())
        state['done'] = True
        fig.canvas.stop_event_loop()

    confirm_button.on_clicked(on_confirm)

    def on_key(event):
        if event.key == 'enter':
            on_confirm()

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show(block=False)
    while not state['done'] and plt.fignum_exists(fig.number):
        fig.canvas.start_event_loop(0.1)

    if not state['done']:
        state['status'] = list(check.get_status())
        print("Window closed without pressing Confirm -- using current checkbox state anyway.")

    if plt.fignum_exists(fig.number):
        plt.close(fig)
        plt.pause(0.01)  # let Qt actually process the close/hide before we return

    selected_labels = [label for label, checked in zip(labels, state['status']) if checked]
    print(f"Selected {len(selected_labels)} sessions: {selected_labels}")

    return selected_labels


def choose_reference_label_popup(selected_labels):
    """
    Numbered-keypress picker for which of the selected sessions should be
    the tracking reference -- same convention review_layer_mismatches_popup
    uses for its 'trust session' choice.

    Parameters
    ----------
    selected_labels : list of str
        From select_sessions_popup (or any list you already have).

    Returns
    -------
    reference_label : str
    """
    n = len(selected_labels)
    if n == 0:
        raise ValueError("selected_labels is empty -- nothing to choose a reference from.")

    fig, ax = plt.subplots(figsize=(7, 1 + 0.4 * n))
    ax.axis('off')
    lines = [f"{i}: {label}" for i, label in enumerate(selected_labels, start=1)]
    ax.text(0.02, 0.5, "\n".join(lines), fontsize=12, va='center', family='monospace')
    fig.suptitle("Press the number of the session to use as reference", fontsize=12, fontweight='bold')
    try:
        fig.canvas.manager.set_window_title('CHOOSE REFERENCE SESSION')
    except Exception:
        pass

    state = {'choice': None}
    valid_keys = {str(k): k - 1 for k in range(1, n + 1)}

    def on_key(event):
        if event.key in valid_keys:
            state['choice'] = valid_keys[event.key]
            fig.canvas.stop_event_loop()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=False)
    while state['choice'] is None and plt.fignum_exists(fig.number):
        fig.canvas.start_event_loop(0.1)
    if plt.fignum_exists(fig.number):
        plt.close(fig)
        plt.pause(0.01)  # let Qt actually process the close/hide before we return

    if state['choice'] is None:
        raise RuntimeError("No reference session chosen (window closed without pressing a number).")

    reference_label = selected_labels[state['choice']]
    print(f"Reference session: {reference_label}")
    return reference_label


def pick_tracking_group_interactively(session_catalog):
    """
    Runs select_sessions_popup then choose_reference_label_popup in
    sequence, handing back exactly what track_and_build_table needs.

    Parameters
    ----------
    session_catalog : dict
        From discover_animal_sessions.

    Returns
    -------
    session_labels : list of str
    reference_label : str
    """
    session_labels = select_sessions_popup(session_catalog)
    if len(session_labels) == 0:
        raise ValueError("No sessions were selected.")
    reference_label = choose_reference_label_popup(session_labels)
    return session_labels, reference_label


# ============================================================
# Function 2.2: per-session loader for tracking
# ============================================================
def load_session_for_tracking(plane0_path):
    """
    Load one session's suite2p output and reconstruct dense footprints +
    centroids for ROI matching. Reimplements TrackROIs_SalineDCZ.py's
    load_suite2p_session + reconstruct_footprints combined into one call,
    since we don't edit the original file.

    Parameters
    ----------
    plane0_path : str
        Path to a .../suite2p/plane0 folder.

    Returns
    -------
    session_data : dict
        'stat': list of iscell-filtered stat dicts
        'mean_img': (Ly, Lx) array
        'Ly', 'Lx': int
        'n_cells': int
        'cell_idx': (n_cells,) iscell-filtered ROI indices into the raw stat.npy
        'footprints': (n_cells, Ly, Lx) float32, lam-weighted
        'centroids': (n_cells, 2) float64, (y, x), lam-weighted
    """
    plane0_path = str(plane0_path)

    stat = np.load(os.path.join(plane0_path, 'stat.npy'), allow_pickle=True)
    iscell = np.load(os.path.join(plane0_path, 'iscell.npy'), allow_pickle=True)
    ops = np.load(os.path.join(plane0_path, 'ops.npy'), allow_pickle=True).item()

    cell_idx = np.where(iscell[:, 0] == 1)[0]
    stat_cells = [stat[i] for i in cell_idx]

    Ly, Lx = ops['Ly'], ops['Lx']
    n_cells = len(stat_cells)

    footprints = np.zeros((n_cells, Ly, Lx), dtype=np.float32)
    centroids = np.zeros((n_cells, 2), dtype=np.float64)

    for i, s in enumerate(stat_cells):
        ypix, xpix, lam = s['ypix'], s['xpix'], s['lam']
        lam_norm = lam / lam.sum()
        footprints[i, ypix, xpix] = lam
        centroids[i, 0] = np.sum(ypix * lam_norm)
        centroids[i, 1] = np.sum(xpix * lam_norm)

    print(f"  Loaded {plane0_path}: {n_cells} cells, image size {Ly}x{Lx}")

    return {
        'stat': stat_cells,
        'mean_img': ops['meanImg'],
        'Ly': Ly,
        'Lx': Lx,
        'n_cells': n_cells,
        'cell_idx': cell_idx,
        'footprints': footprints,
        'centroids': centroids,
    }


# ============================================================
# Function 2.3: core flexible matching step
# ============================================================
def _align_session_group_to_reference(sessions, reference_label):
    """
    FOV registration (translation only) of every session's mean image
    against the reference's, via phase_cross_correlation.

    Returns
    -------
    shifts : dict
        {label: (dy, dx)} in pixels. Reference itself is (0.0, 0.0).
    """
    if reference_label not in sessions:
        raise ValueError(f"reference_label '{reference_label}' not in sessions "
                          f"(available: {list(sessions.keys())})")

    ref_img = sessions[reference_label]['mean_img']
    shifts = {}

    for label, sd in sessions.items():
        if label == reference_label:
            shifts[label] = (0.0, 0.0)
            print(f"  {label}: reference (0, 0)")
            continue

        shift_yx, error, diffphase = phase_cross_correlation(
            ref_img, sd['mean_img'], upsample_factor=10
        )
        shifts[label] = (shift_yx[0], shift_yx[1])
        print(f"  {label}: shift = ({shift_yx[0]:.2f}, {shift_yx[1]:.2f}) pixels")

    return shifts


def _match_two_sessions(session_a, session_b, shift_a, shift_b,
                         max_distance_um=15.0, min_correlation=0.3,
                         microns_per_pixel=MICRONS_PER_PIXEL):
    """
    Match ROIs from session_a (typically the reference) to session_b:
    candidate pairs by shifted-centroid distance, footprint correlation on
    those candidates, then Hungarian assignment on the surviving pairs.
    Same logic as TrackROIs_SalineDCZ.py's match_rois_pairwise, generalized
    to take pre-shifted centroids via arbitrary shift_a/shift_b rather than
    assuming session_a is always the unshifted reference.

    Parameters
    ----------
    session_a, session_b : dict
        From load_session_for_tracking.
    shift_a, shift_b : tuple of float
        (dy, dx) for each session, from _align_session_group_to_reference.
    max_distance_um, min_correlation : float
    microns_per_pixel : float

    Returns
    -------
    matches : list of dict
        Each: {'idx_a', 'idx_b', 'distance_um', 'correlation'}.
    """
    centroids_a = session_a['centroids'] + np.array(shift_a)
    centroids_b = session_b['centroids'] + np.array(shift_b)

    dist_matrix = cdist(centroids_a, centroids_b, metric='euclidean') * microns_per_pixel

    candidate_pairs = [(i, j) for i in range(dist_matrix.shape[0])
                       for j in range(dist_matrix.shape[1])
                       if dist_matrix[i, j] <= max_distance_um]

    if len(candidate_pairs) == 0:
        return []

    correlations = {}
    for idx_a, idx_b in candidate_pairs:
        fp_a = ndi_shift(session_a['footprints'][idx_a], shift=shift_a, mode='constant', cval=0)
        fp_b = ndi_shift(session_b['footprints'][idx_b], shift=shift_b, mode='constant', cval=0)

        mask = (fp_a > 0) | (fp_b > 0)
        if mask.sum() < 5:
            correlations[(idx_a, idx_b)] = 0.0
            continue

        vals_a, vals_b = fp_a[mask], fp_b[mask]
        if vals_a.std() < 1e-10 or vals_b.std() < 1e-10:
            correlations[(idx_a, idx_b)] = 0.0
            continue

        correlations[(idx_a, idx_b)] = np.corrcoef(vals_a, vals_b)[0, 1]

    valid_pairs = [(i, j) for (i, j) in candidate_pairs
                   if correlations.get((i, j), 0) >= min_correlation]
    if len(valid_pairs) == 0:
        return []

    unique_a = sorted(set(i for i, j in valid_pairs))
    unique_b = sorted(set(j for i, j in valid_pairs))
    map_a = {idx: k for k, idx in enumerate(unique_a)}
    map_b = {idx: k for k, idx in enumerate(unique_b)}

    cost_matrix = np.full((len(unique_a), len(unique_b)), 1e6)
    for i, j in valid_pairs:
        cost_matrix[map_a[i], map_b[j]] = dist_matrix[i, j]

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 1e6:
            idx_a, idx_b = unique_a[r], unique_b[c]
            matches.append({
                'idx_a': idx_a,
                'idx_b': idx_b,
                'distance_um': dist_matrix[idx_a, idx_b],
                'correlation': correlations[(idx_a, idx_b)],
            })

    return matches


def track_session_group(sessions, reference_label, max_distance_um=15.0,
                         min_correlation=0.3, microns_per_pixel=MICRONS_PER_PIXEL):
    """
    Register + match ROIs across an arbitrary caller-chosen group of
    sessions, against a caller-chosen reference session in that group.

    Parameters
    ----------
    sessions : dict
        {label: session_data}, from load_session_for_tracking. Any size --
        typically 2-3 for one scientific comparison.
    reference_label : str
        Must be a key in sessions.
    max_distance_um, min_correlation : float
    microns_per_pixel : float

    Returns
    -------
    tracking_result : dict
        'registration_matrix': (n_ref_cells, n_sessions) int array,
            columns ordered by labels_order, -1 = not found in that session.
        'labels_order': list of str
        'reference_label': str
        'shifts': {label: (dy, dx)}
        'matches': {label: [match dicts]} (non-reference labels only)
        'sessions': the input sessions dict, passed through
    """
    print(f"Aligning {len(sessions)} sessions to reference '{reference_label}'...")
    shifts = _align_session_group_to_reference(sessions, reference_label)

    labels_order = list(sessions.keys())
    ref_col = labels_order.index(reference_label)
    n_ref_cells = sessions[reference_label]['n_cells']

    registration_matrix = np.full((n_ref_cells, len(labels_order)), -1, dtype=int)
    registration_matrix[:, ref_col] = np.arange(n_ref_cells)

    matches = {}
    print("\nMatching non-reference sessions...")
    for label in labels_order:
        if label == reference_label:
            continue

        match_list = _match_two_sessions(
            sessions[reference_label], sessions[label],
            shifts[reference_label], shifts[label],
            max_distance_um=max_distance_um, min_correlation=min_correlation,
            microns_per_pixel=microns_per_pixel
        )
        matches[label] = match_list

        col = labels_order.index(label)
        for m in match_list:
            registration_matrix[m['idx_a'], col] = m['idx_b']

        print(f"  {reference_label} vs {label}: {len(match_list)} matches "
              f"(out of {n_ref_cells} reference cells)")

    sessions_per_cell = np.sum(registration_matrix >= 0, axis=1)
    print(f"\nTracked in all {len(labels_order)} sessions: "
          f"{np.sum(sessions_per_cell == len(labels_order))} / {n_ref_cells} reference cells")

    return {
        'registration_matrix': registration_matrix,
        'labels_order': labels_order,
        'reference_label': reference_label,
        'shifts': shifts,
        'matches': matches,
        'sessions': sessions,
    }


# ============================================================
# Function 2.4: auto-accept/auto-reject/manual-review categorization
# ============================================================
def filter_and_flag_matches(tracking_result, max_footprint_pixels=500,
                             auto_accept_corr=0.7, auto_accept_dist=5.0,
                             microns_per_pixel=MICRONS_PER_PIXEL):
    """
    Of the cells tracked across every session in the group, sorts each into:
    - 'auto_reject' -- footprint size in any session exceeds
      max_footprint_pixels (a segmentation artifact/merged-ROI red flag)
    - 'auto_accept' -- footprint correlation and centroid distance vs. the
      reference are within threshold in every non-reference session
    - 'manual_review' -- everything else

    Same logic as TrackROIs_SalineDCZ.py's filter_registration_matrix, but
    generalized from a hardcoded 2-session pair to "worst across however
    many non-reference sessions are in this group."

    Parameters
    ----------
    tracking_result : dict
        From track_session_group.
    max_footprint_pixels : int
    auto_accept_corr : float
    auto_accept_dist : float
    microns_per_pixel : float

    Returns
    -------
    filtered_matrix : numpy.ndarray
    cell_categories : dict
        {row: 'auto_accept' | 'auto_reject' | 'manual_review'}
    filter_report : dict
    """
    registration_matrix = tracking_result['registration_matrix']
    labels_order = tracking_result['labels_order']
    reference_label = tracking_result['reference_label']
    shifts = tracking_result['shifts']
    sessions = tracking_result['sessions']

    ref_col = labels_order.index(reference_label)
    non_ref_labels = [l for l in labels_order if l != reference_label]

    mask = np.all(registration_matrix >= 0, axis=1)
    candidate_rows = np.where(mask)[0]

    filtered_matrix = registration_matrix.copy()
    cell_categories = {}
    reject_details = []
    accept_details = []

    print(f"Evaluating {len(candidate_rows)} cells tracked across all "
          f"{len(labels_order)} sessions...")

    for row in candidate_rows:
        # --- Footprint-size artifact check, every session ---
        is_artifact = False
        for col, label in enumerate(labels_order):
            roi_idx = registration_matrix[row, col]
            fp_size = np.sum(sessions[label]['footprints'][roi_idx] > 0)
            if fp_size > max_footprint_pixels:
                is_artifact = True
                reject_details.append({'row': int(row), 'label': label,
                                        'roi_idx': int(roi_idx), 'fp_size': int(fp_size)})
                break

        if is_artifact:
            filtered_matrix[row, :] = -1
            cell_categories[row] = 'auto_reject'
            continue

        # --- Worst correlation/distance vs reference, every non-reference session ---
        ref_roi = registration_matrix[row, ref_col]
        fp_ref = ndi_shift(sessions[reference_label]['footprints'][ref_roi],
                            shift=shifts[reference_label], mode='constant', cval=0)
        cent_ref = sessions[reference_label]['centroids'][ref_roi] + np.array(shifts[reference_label])

        all_high_quality = True
        worst_corr, worst_dist = 1.0, 0.0

        for label in non_ref_labels:
            col = labels_order.index(label)
            roi_idx = registration_matrix[row, col]

            cent_other = sessions[label]['centroids'][roi_idx] + np.array(shifts[label])
            dist_um = np.linalg.norm(cent_ref - cent_other) * microns_per_pixel

            fp_other = ndi_shift(sessions[label]['footprints'][roi_idx],
                                  shift=shifts[label], mode='constant', cval=0)
            union_mask = (fp_ref > 0) | (fp_other > 0)
            if union_mask.sum() < 5:
                all_high_quality = False
                break

            vals_a, vals_b = fp_ref[union_mask], fp_other[union_mask]
            if vals_a.std() < 1e-10 or vals_b.std() < 1e-10:
                all_high_quality = False
                break

            corr = np.corrcoef(vals_a, vals_b)[0, 1]
            worst_corr = min(worst_corr, corr)
            worst_dist = max(worst_dist, dist_um)

            if corr < auto_accept_corr or dist_um > auto_accept_dist:
                all_high_quality = False
                break

        if all_high_quality:
            cell_categories[row] = 'auto_accept'
            accept_details.append({'row': int(row), 'worst_corr': worst_corr, 'worst_dist': worst_dist})
        else:
            cell_categories[row] = 'manual_review'

    n_accept = sum(1 for v in cell_categories.values() if v == 'auto_accept')
    n_reject = sum(1 for v in cell_categories.values() if v == 'auto_reject')
    n_manual = sum(1 for v in cell_categories.values() if v == 'manual_review')

    print(f"\n  Auto-accept:   {n_accept}")
    print(f"  Auto-reject:   {n_reject}  (footprint > {max_footprint_pixels}px in some session)")
    print(f"  Manual review: {n_manual}")

    filter_report = {
        'n_candidates': len(candidate_rows),
        'n_auto_accept': n_accept,
        'n_auto_reject': n_reject,
        'n_manual_review': n_manual,
        'reject_details': reject_details,
        'accept_details': accept_details,
    }

    return filtered_matrix, cell_categories, filter_report


# ============================================================
# Function 2.5: interactive keyboard review of the manual_review bucket
# ============================================================
def plot_cell_across_group(tracking_result, cell_row, crop_size=40,
                            microns_per_pixel=MICRONS_PER_PIXEL):
    """
    One figure for a single tracked cell: a footprint-crop panel per
    session in the group, plus an "all overlaid" panel colored by session.

    Parameters
    ----------
    tracking_result : dict
        From track_session_group.
    cell_row : int
        Row index into registration_matrix (a reference-session ROI index).
    crop_size : int
        Half-width of the crop window, in pixels.
    microns_per_pixel : float

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    registration_matrix = tracking_result['registration_matrix']
    labels_order = tracking_result['labels_order']
    reference_label = tracking_result['reference_label']
    shifts = tracking_result['shifts']
    sessions = tracking_result['sessions']

    ref_col = labels_order.index(reference_label)
    ref_roi = registration_matrix[cell_row, ref_col]
    fp_ref = ndi_shift(sessions[reference_label]['footprints'][ref_roi],
                        shift=shifts[reference_label], mode='constant', cval=0)
    cent_ref = sessions[reference_label]['centroids'][ref_roi] + np.array(shifts[reference_label])

    corrs, dists, all_cents = {}, {}, []
    for label in labels_order:
        col = labels_order.index(label)
        roi_idx = registration_matrix[cell_row, col]
        cent = sessions[label]['centroids'][roi_idx] + np.array(shifts[label])
        all_cents.append(cent)
        dists[label] = np.linalg.norm(cent_ref - cent) * microns_per_pixel

        if label == reference_label:
            corrs[label] = 1.0
            continue

        fp_other = ndi_shift(sessions[label]['footprints'][roi_idx],
                              shift=shifts[label], mode='constant', cval=0)
        union_mask = (fp_ref > 0) | (fp_other > 0)
        if union_mask.sum() < 5:
            corrs[label] = 0.0
            continue
        vals_a, vals_b = fp_ref[union_mask], fp_other[union_mask]
        if vals_a.std() < 1e-10 or vals_b.std() < 1e-10:
            corrs[label] = 0.0
            continue
        corrs[label] = np.corrcoef(vals_a, vals_b)[0, 1]

    all_cents = np.array(all_cents)
    mid_y, mid_x = int(np.mean(all_cents[:, 0])), int(np.mean(all_cents[:, 1]))

    Ly, Lx = sessions[reference_label]['Ly'], sessions[reference_label]['Lx']
    y_min, y_max = max(0, mid_y - crop_size), min(Ly, mid_y + crop_size)
    x_min, x_max = max(0, mid_x - crop_size), min(Lx, mid_x + crop_size)

    n_sessions = len(labels_order)
    fig, axes = plt.subplots(1, n_sessions + 1, figsize=(4 * (n_sessions + 1), 4))

    footprints_cropped = []
    for i, label in enumerate(labels_order):
        col = labels_order.index(label)
        roi_idx = registration_matrix[cell_row, col]

        fp = ndi_shift(sessions[label]['footprints'][roi_idx],
                        shift=shifts[label], mode='constant', cval=0)
        mean_shifted = ndi_shift(sessions[label]['mean_img'],
                                  shift=shifts[label], mode='constant', cval=0)

        # ndi_shift's default cubic-spline interpolation can introduce tiny
        # negative overshoot near sharp edges even for a non-negative input
        # (footprint values are all >= 0) -- clip it out before normalizing,
        # so imshow doesn't warn about out-of-range RGBA values below.
        fp_crop = np.clip(fp[y_min:y_max, x_min:x_max], 0, None)
        mean_crop = mean_shifted[y_min:y_max, x_min:x_max]
        footprints_cropped.append(fp_crop)

        fp_norm = fp_crop / fp_crop.max() if fp_crop.max() > 0 else fp_crop
        axes[i].imshow(mean_crop, cmap='gray')
        overlay = np.zeros((*fp_crop.shape, 4))
        overlay[:, :, 1] = fp_norm
        overlay[:, :, 3] = fp_norm * 0.6
        axes[i].imshow(overlay)

        if label == reference_label:
            axes[i].set_title(f"{label} (reference)\nROI {roi_idx}", fontsize=9)
        else:
            axes[i].set_title(f"{label}\nROI {roi_idx}\n"
                               f"corr={corrs[label]:.2f}, dist={dists[label]:.1f}um", fontsize=9)
        axes[i].axis('off')

    colors = plt.cm.hsv(np.linspace(0, 0.85, n_sessions))
    ref_mean_shifted = ndi_shift(sessions[reference_label]['mean_img'],
                                  shift=shifts[reference_label], mode='constant', cval=0)
    mean_crop_ref = ref_mean_shifted[y_min:y_max, x_min:x_max]

    axes[n_sessions].imshow(mean_crop_ref, cmap='gray')
    overlay_all = np.zeros((*footprints_cropped[0].shape, 4))
    for j, fp_crop in enumerate(footprints_cropped):
        fp_norm = fp_crop / fp_crop.max() if fp_crop.max() > 0 else fp_crop
        mask = fp_norm > 0.1
        overlay_all[mask, 0] += colors[j, 0] * fp_norm[mask]
        overlay_all[mask, 1] += colors[j, 1] * fp_norm[mask]
        overlay_all[mask, 2] += colors[j, 2] * fp_norm[mask]
        overlay_all[mask, 3] = np.maximum(overlay_all[mask, 3], fp_norm[mask] * 0.6)
    overlay_all[:, :, :3] = np.clip(overlay_all[:, :, :3], 0, 1)
    axes[n_sessions].imshow(overlay_all)

    non_ref = [l for l in labels_order if l != reference_label]
    if non_ref:
        min_corr = min(corrs[l] for l in non_ref)
        max_dist = max(dists[l] for l in non_ref)
        axes[n_sessions].set_title(f"All overlaid\nmin corr={min_corr:.2f}\n"
                                    f"max dist={max_dist:.1f}um", fontsize=9)
    else:
        axes[n_sessions].set_title("All overlaid", fontsize=9)
    axes[n_sessions].axis('off')

    fig.suptitle(f"Cell row {cell_row} (reference: {reference_label} ROI {ref_roi})",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def manual_review_matches(tracking_result, cell_categories, crop_size=40,
                           microns_per_pixel=MICRONS_PER_PIXEL):
    """
    Interactive keyboard review for every row categorized 'manual_review'.
    'a'/Right = accept, 'r'/'x' = reject, 'b'/Left = back, 'q' = quit early.

    Waits for each keypress via fig.canvas.start_event_loop() polling
    (opens/closes a fresh figure per cell) rather than a blocking
    plt.show(), which raced with Jupyter's own Qt event-loop integration in
    earlier testing and could render a stale frame.

    Parameters
    ----------
    tracking_result : dict
        From track_session_group.
    cell_categories : dict
        From filter_and_flag_matches.
    crop_size : int
    microns_per_pixel : float

    Returns
    -------
    verified_matrix : numpy.ndarray
        registration_matrix restricted to auto_accept rows plus
        manually-accepted rows -- auto_reject, manually-rejected, AND any
        manual_review row you didn't get to before quitting are all
        excluded (not "verified" just because you ran out of time).
    decisions : dict
        {row: 'accept' | 'reject'} for every row actually reviewed.
    """
    manual_rows = sorted([row for row, cat in cell_categories.items() if cat == 'manual_review'])
    total = len(manual_rows)

    print(f"Manual review: {total} cells")
    print("Controls: 'a'/Right = accept, 'r'/'x' = reject, 'b'/Left = back, 'q' = quit\n")

    decisions = {}
    idx = 0

    while 0 <= idx < total:
        row = manual_rows[idx]
        fig = plot_cell_across_group(tracking_result, row, crop_size=crop_size,
                                      microns_per_pixel=microns_per_pixel)

        status = decisions.get(row, None)
        status_str = "NOT REVIEWED" if status is None else status.upper()
        fig.suptitle(f"Manual Review {idx+1}/{total} (row {row}) | Status: {status_str}\n"
                     "Click figure, then 'a'=accept  'r'=reject  'b'=back  'q'=quit",
                     fontsize=11, fontweight='bold')

        action = {'value': None}

        def on_key(event):
            if event.key in ('a', 'right'):
                action['value'] = 'accept'
                fig.canvas.stop_event_loop()
            elif event.key in ('r', 'x'):
                action['value'] = 'reject'
                fig.canvas.stop_event_loop()
            elif event.key in ('b', 'left'):
                action['value'] = 'back'
                fig.canvas.stop_event_loop()
            elif event.key == 'q':
                action['value'] = 'quit'
                fig.canvas.stop_event_loop()

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show(block=False)

        while action['value'] is None and plt.fignum_exists(fig.number):
            fig.canvas.start_event_loop(0.1)

        if plt.fignum_exists(fig.number):
            plt.close(fig)

        if action['value'] in ('accept', 'reject'):
            decisions[row] = action['value']
            idx += 1
        elif action['value'] == 'back':
            idx = max(0, idx - 1)
        else:  # 'quit' or window closed without a keypress
            print("\nQuitting review.")
            break

    # "Verified" = auto_accept rows, plus manually-accepted rows. Everything
    # else (auto_reject, manually-rejected, and any manual_review row not
    # reached before quitting) is excluded -- being unreviewed is not the
    # same as being accepted.
    verified_matrix = tracking_result['registration_matrix'].copy()
    for row, category in cell_categories.items():
        if category == 'auto_reject':
            verified_matrix[row, :] = -1
        elif category == 'manual_review':
            if decisions.get(row) != 'accept':
                verified_matrix[row, :] = -1
        # 'auto_accept' rows: left as-is (already valid in the matrix)

    n_accept = sum(1 for d in decisions.values() if d == 'accept')
    n_reject = sum(1 for d in decisions.values() if d == 'reject')
    n_unreviewed = total - len(decisions)

    print(f"\n  Manually accepted: {n_accept}")
    print(f"  Manually rejected: {n_reject}")
    print(f"  Unreviewed:        {n_unreviewed}  (excluded from verified_matrix, not accepted)")

    return verified_matrix, decisions


# ============================================================
# Function 2.6: persist / reload one session-group's verified tracking result
# ============================================================
def save_tracking_group(tracking_result, verified_matrix, decisions, filter_report,
                         plane0_paths, group_name, save_dir=None):
    """
    Save one session-group's verified tracking result to HDF5, as
    '{group_name}_tracking_results.h5' in a 'TrackedGroups/' subfolder of
    the reference session's TSeries directory -- mirroring where
    TrackROIs_SalineDCZ.py saves roi_tracking_results.h5 (a 'TrackedROIs/'
    subfolder next to the sessions).

    Saves the verified matrix (post auto-filter + manual review), the
    original unfiltered matrix (for audit), labels_order/reference_label,
    per-session plane0_path (so downstream steps can reload sessions
    without re-discovering them), shifts, and the decisions dict from
    manual review.

    Parameters
    ----------
    tracking_result : dict
        From track_session_group.
    verified_matrix : numpy.ndarray
        Post auto-filter + manual review (from manual_review_matches, or
        just filtered_matrix if you skipped manual review).
    decisions : dict
        From manual_review_matches (may be empty).
    filter_report : dict
        From filter_and_flag_matches.
    plane0_paths : dict
        {label: plane0_path}.
    group_name : str
        e.g. "BaselineDay5_vs_DCZ1".
    save_dir : str, optional
        Defaults to a 'TrackedGroups' folder next to the reference session's
        TSeries directory.

    Returns
    -------
    save_path : str
    """
    labels_order = tracking_result['labels_order']
    reference_label = tracking_result['reference_label']
    shifts = tracking_result['shifts']
    registration_matrix = tracking_result['registration_matrix']

    if save_dir is None:
        ref_tseries_dir = os.path.dirname(os.path.dirname(str(plane0_paths[reference_label])))
        save_dir = os.path.join(ref_tseries_dir, 'TrackedGroups')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'{group_name}_tracking_results.h5')

    with h5py.File(save_path, 'w') as f:
        f.create_dataset('verified_matrix', data=verified_matrix)
        f.create_dataset('original_matrix', data=registration_matrix)

        f.attrs['group_name'] = group_name
        f.attrs['labels_order'] = labels_order
        f.attrs['reference_label'] = reference_label

        shift_grp = f.create_group('shifts')
        for label, (dy, dx) in shifts.items():
            shift_grp.attrs[label] = [dy, dx]

        plane0_grp = f.create_group('plane0_paths')
        for label in labels_order:
            plane0_grp.attrs[label] = str(plane0_paths[label])

        if decisions:
            dec_grp = f.create_group('decisions')
            rows = list(decisions.keys())
            vals = [decisions[r] for r in rows]
            dec_grp.create_dataset('rows', data=np.array(rows, dtype=int))
            dec_grp.create_dataset('decisions', data=np.array(vals, dtype='S10'))

        report_grp = f.create_group('filter_report')
        for k in ('n_candidates', 'n_auto_accept', 'n_auto_reject', 'n_manual_review'):
            if k in filter_report:
                report_grp.attrs[k] = filter_report[k]

    n_tracked = np.sum(np.all(verified_matrix >= 0, axis=1))
    print(f"Saved tracking group '{group_name}' -> {save_path}")
    print(f"  Sessions: {labels_order}  (reference: {reference_label})")
    print(f"  Final tracked cells: {n_tracked}")

    return save_path


def _decode(x):
    """h5py attrs may come back as bytes depending on version -- normalize to str."""
    return x.decode() if isinstance(x, bytes) else x


def load_tracking_group(save_path):
    """
    Reload a tracking group saved by save_tracking_group.

    Returns
    -------
    result : dict with keys:
        verified_matrix, original_matrix, labels_order, reference_label,
        shifts, plane0_paths, decisions, filter_report.
    """
    with h5py.File(save_path, 'r') as f:
        verified_matrix = f['verified_matrix'][:]
        original_matrix = f['original_matrix'][:]
        labels_order = [_decode(l) for l in f.attrs['labels_order']]
        reference_label = _decode(f.attrs['reference_label'])

        shifts = {label: tuple(f['shifts'].attrs[label]) for label in labels_order}
        plane0_paths = {label: _decode(f['plane0_paths'].attrs[label]) for label in labels_order}

        decisions = {}
        if 'decisions' in f:
            rows = f['decisions']['rows'][:]
            vals = [_decode(v) for v in f['decisions']['decisions'][:]]
            decisions = {int(r): v for r, v in zip(rows, vals)}

        filter_report = {k: v for k, v in f['filter_report'].attrs.items()}

    return {
        'verified_matrix': verified_matrix,
        'original_matrix': original_matrix,
        'labels_order': labels_order,
        'reference_label': reference_label,
        'shifts': shifts,
        'plane0_paths': plane0_paths,
        'decisions': decisions,
        'filter_report': filter_report,
    }


# ============================================================
# Dev checkpoint utility (optional convenience, not part of the core
# pipeline) -- save/reload a tracking+filter result (pre-manual-review)
# so a kernel restart mid-way through a long manual review doesn't force
# redoing alignment/matching/filtering. Skips pickling
# tracking_result['sessions'] (the large dense footprint arrays) -- saves
# each session's plane0_path instead and reconstructs sessions fresh via
# load_session_for_tracking on load, since that's fast and keeps the
# checkpoint file small.
# ============================================================
def save_tracking_checkpoint(tracking_result, cell_categories, filter_report,
                              plane0_paths, checkpoint_path):
    """Save the lightweight parts of a tracking run to disk (see module note above)."""
    checkpoint = {
        'registration_matrix': tracking_result['registration_matrix'],
        'labels_order': tracking_result['labels_order'],
        'reference_label': tracking_result['reference_label'],
        'shifts': tracking_result['shifts'],
        'matches': tracking_result['matches'],
        'cell_categories': cell_categories,
        'filter_report': filter_report,
        'plane0_paths': {label: str(plane0_paths[label]) for label in tracking_result['labels_order']},
    }
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Saved checkpoint -> {checkpoint_path}")


def load_tracking_checkpoint(checkpoint_path, reload_sessions=True):
    """Reload a checkpoint saved by save_tracking_checkpoint (see module note above)."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    sessions = {}
    if reload_sessions:
        print("Reloading sessions from suite2p output...")
        for label, plane0_path in checkpoint['plane0_paths'].items():
            sessions[label] = load_session_for_tracking(plane0_path)

    tracking_result = {
        'registration_matrix': checkpoint['registration_matrix'],
        'labels_order': checkpoint['labels_order'],
        'reference_label': checkpoint['reference_label'],
        'shifts': checkpoint['shifts'],
        'matches': checkpoint['matches'],
        'sessions': sessions,
    }

    print(f"Loaded checkpoint from {checkpoint_path}")
    return tracking_result, checkpoint['cell_categories'], checkpoint['filter_report']


# ============================================================
# Function 2.7: join the verified matrix with Phase 1's layer labels
# ============================================================
def find_layer_curve_path(plane0_path, prefer_averaged=True):
    """
    Locate a session's Phase 1 layer-curve-results file, given its
    suite2p/plane0 path -- same glob-based discovery convention as
    MergeTrackedLayerSMI.py's find_smi_results_path.

    Two Phase 1 outputs can coexist per session: the retired
    independent-click version's output (*_layer_curve_results.h5) and the
    current 1.LayerAssignment_Curve.py's chained-registration/averaged
    output (*_layer_curve_results_averaged.h5, kept under that suffix even
    now that it's the only Phase 1 script, specifically so it never
    overwrites the independent-click results already computed for real) --
    the two glob patterns below are naturally disjoint (the averaged
    filename doesn't end in '_layer_curve_results.h5'), so this never
    confuses the two. Prefers the averaged variant by default since that's
    the current script's output; pass prefer_averaged=False to force the
    retired version's if you need to compare against it.
    """
    tseries_dir = os.path.dirname(os.path.dirname(str(plane0_path)))
    averaged_matches = glob.glob(os.path.join(tseries_dir, '*_layer_curve_results_averaged.h5'))
    independent_matches = glob.glob(os.path.join(tseries_dir, '*_layer_curve_results.h5'))

    if prefer_averaged and averaged_matches:
        matches = averaged_matches
    elif independent_matches:
        matches = independent_matches
    elif averaged_matches:
        matches = averaged_matches
    else:
        raise FileNotFoundError(f"No *_layer_curve_results(_averaged).h5 found in {tseries_dir} "
                                 "-- has Phase 1 been run for this session?")

    if len(matches) > 1:
        print(f"WARNING: multiple matches in {tseries_dir}, using {matches[0]}")
    return matches[0]


def _load_layer_of_cell(layer_curve_path):
    """
    Load the {roi_idx: layer_name} mapping from a Phase 1
    *_layer_curve_results.h5 file, keyed by POSITION within the
    iscell-filtered cell list (0-based) -- matching Phase 2's own roi_idx
    convention in registration_matrix/verified_matrix (built from
    load_session_for_tracking's footprints[i]/centroids[i], which are
    indexed by enumerate() position, not by raw suite2p ROI index).

    BUG FIX (found during real-data testing): this used to key the dict by
    the file's own 'cell_idx' dataset (the RAW suite2p stat.npy index, e.g.
    3, 17, 108... after most raw detections get filtered out by iscell)
    instead of by position. Since Phase 2's roi_idx is a position, not a
    raw index, that lookup was wrong for nearly every cell -- coincidental
    collisions gave silently WRONG layer labels, and non-collisions gave
    spurious None values. Both Phase 1 and Phase 2 build their
    iscell-filtered lists the same way (np.where(iscell[:,0]==1)[0]) from
    the same iscell.npy, so position i refers to the same physical cell in
    both -- that's the correct join key.
    """
    with h5py.File(layer_curve_path, 'r') as f:
        layer_codes = f['layer_codes'][:]
        layer_names = tuple(n.decode() if isinstance(n, bytes) else n
                             for n in f['layer_names'][:])
    return {
        position: layer_names[code]
        for position, code in enumerate(layer_codes) if code >= 0
    }


def _worst_corr_dist_for_row(tracking_result, row, microns_per_pixel=MICRONS_PER_PIXEL):
    """
    Worst footprint correlation / largest centroid distance (vs reference)
    across all non-reference sessions, for one tracked cell -- same
    computation filter_and_flag_matches uses, factored out for reuse.
    """
    registration_matrix = tracking_result['registration_matrix']
    labels_order = tracking_result['labels_order']
    reference_label = tracking_result['reference_label']
    shifts = tracking_result['shifts']
    sessions = tracking_result['sessions']

    ref_col = labels_order.index(reference_label)
    ref_roi = registration_matrix[row, ref_col]
    fp_ref = ndi_shift(sessions[reference_label]['footprints'][ref_roi],
                        shift=shifts[reference_label], mode='constant', cval=0)
    cent_ref = sessions[reference_label]['centroids'][ref_roi] + np.array(shifts[reference_label])

    worst_corr, worst_dist = 1.0, 0.0
    for label in labels_order:
        if label == reference_label:
            continue
        col = labels_order.index(label)
        roi_idx = registration_matrix[row, col]

        cent_other = sessions[label]['centroids'][roi_idx] + np.array(shifts[label])
        dist_um = np.linalg.norm(cent_ref - cent_other) * microns_per_pixel

        fp_other = ndi_shift(sessions[label]['footprints'][roi_idx],
                              shift=shifts[label], mode='constant', cval=0)
        union_mask = (fp_ref > 0) | (fp_other > 0)
        if union_mask.sum() < 5:
            corr = 0.0
        else:
            vals_a, vals_b = fp_ref[union_mask], fp_other[union_mask]
            corr = 0.0 if (vals_a.std() < 1e-10 or vals_b.std() < 1e-10) else np.corrcoef(vals_a, vals_b)[0, 1]

        worst_corr = min(worst_corr, corr)
        worst_dist = max(worst_dist, dist_um)

    return worst_corr, worst_dist


def build_master_cell_table(tracking_result, verified_matrix, layer_curve_paths,
                             microns_per_pixel=MICRONS_PER_PIXEL):
    """
    Join a verified tracking-group registration matrix with each session's
    Phase 1 layer labels into one tidy table.

    Parameters
    ----------
    tracking_result : dict
        From track_session_group.
    verified_matrix : numpy.ndarray
        Post auto-filter + manual review.
    layer_curve_paths : dict
        {label: path to that session's *_layer_curve_results.h5}.
    microns_per_pixel : float

    Returns
    -------
    df : pandas.DataFrame
        One row per verified tracked cell: global_cell_id, then per-session
        roi_idx_<label>/layer_<label>, plus worst_corr, worst_dist_um.
    """
    labels_order = tracking_result['labels_order']

    layer_lookup = {label: _load_layer_of_cell(layer_curve_paths[label]) for label in labels_order}

    mask = np.all(verified_matrix >= 0, axis=1)
    tracked_rows = np.where(mask)[0]
    print(f"Building master table for {len(tracked_rows)} verified tracked cells...")

    records = []
    for row in tracked_rows:
        record = {'global_cell_id': int(row)}
        for col, label in enumerate(labels_order):
            roi_idx = int(verified_matrix[row, col])
            record[f'roi_idx_{label}'] = roi_idx
            record[f'layer_{label}'] = layer_lookup[label].get(roi_idx, None)

        worst_corr, worst_dist = _worst_corr_dist_for_row(tracking_result, row, microns_per_pixel)
        record['worst_corr'] = worst_corr
        record['worst_dist_um'] = worst_dist
        records.append(record)

    df = pd.DataFrame.from_records(records)

    layer_cols = [c for c in df.columns if c.startswith('layer_')]
    n_layer_mismatch = df[layer_cols].nunique(axis=1).gt(1).sum()
    print(f"  {len(df)} cells in master table.")
    print(f"  {n_layer_mismatch} cells have differing layer labels across sessions "
          f"(flagged for Function 2.8, flag_layer_mismatches).")

    return df


# ============================================================
# Fix -- save_master_cell_table / load_master_cell_table
#
# track_and_build_table's save_tracking_group call only persists the raw
# tracking matrix/decisions/filter_report -- master_df (the joined
# tracked-cell + layer table Track A's join_smi_to_master_table actually
# needs) was only ever returned in-memory, never written to disk. Every
# time Track A work resumes, this would force re-running tracking from
# scratch just to regenerate a table that was already built and reviewed
# once. Saves master_df alongside save_tracking_group's output, plus the
# mismatch-review resolutions dict (if any), which was being lost too.
# ============================================================

def save_master_cell_table(master_df, group_name, save_dir, resolutions=None):
    """
    Save one tracking group's master_df to '{group_name}_master_table.csv'
    in save_dir, alongside save_tracking_group's
    '{group_name}_tracking_results.h5'. Also saves the mismatch-review
    resolutions (if any) as '{group_name}_mismatch_resolutions.json'.

    Parameters
    ----------
    master_df : pandas.DataFrame
        From build_master_cell_table (optionally through flag_layer_mismatches).
    group_name : str
    save_dir : str
        Same directory save_tracking_group used (its return value's dirname).
    resolutions : dict, optional
        From review_layer_mismatches_popup.

    Returns
    -------
    saved_paths : dict
        {'table': path, 'resolutions': path or None}.
    """
    os.makedirs(save_dir, exist_ok=True)

    table_path = os.path.join(save_dir, f'{group_name}_master_table.csv')
    master_df.to_csv(table_path, index=False)
    print(f"Saved master table -> {table_path}")

    resolutions_path = None
    if resolutions:
        resolutions_path = os.path.join(save_dir, f'{group_name}_mismatch_resolutions.json')
        serializable = {str(k): v for k, v in resolutions.items()}
        with open(resolutions_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved mismatch resolutions -> {resolutions_path}")

    return {'table': table_path, 'resolutions': resolutions_path}


def load_master_cell_table(save_dir, group_name):
    """
    Reload one tracking group's saved master table -- undoes
    save_master_cell_table's to_csv. Means Track A's later revisit doesn't
    need to re-run tracking from scratch just to get master_df back --
    exactly the gap this fix closes.

    Parameters
    ----------
    save_dir : str
    group_name : str

    Returns
    -------
    master_df : pandas.DataFrame
    """
    table_path = os.path.join(save_dir, f'{group_name}_master_table.csv')
    master_df = pd.read_csv(table_path)
    print(f"Loaded master table <- {table_path} ({len(master_df)} rows)")
    return master_df


# ============================================================
# Function 2.8: flag cells whose layer label disagrees across sessions
# ============================================================
def flag_layer_mismatches(master_df):
    """
    Identify tracked cells whose Phase 1 layer label differs across
    sessions in the group. Generalizes MergeTrackedLayerSMI.py's pairwise
    n_layer_changes check to however many layer_<label> columns are in the
    master table (2-3+, not just 2).

    Note: nunique ignores None/NaN by default, so a cell missing a layer
    label in one session (shouldn't happen given Phase 1's exhaustive
    assignment, but not impossible) could mask a real mismatch between its
    other sessions. Not fixed since it shouldn't occur in practice.

    On real JSY090 data (Day5/DCZ1 triplet), this legitimately caught a
    ~66% mismatch rate on the first pass -- which traced back to Phase 1's
    boundary curve, not to anything in this function or in Phase 2's
    tracking. See this module's docstring. After that root cause was fixed
    in Phase 1, plus a roi_idx indexing bug fixed in _load_layer_of_cell
    above, the same group came out at 3/119 -- i.e. this function and
    Function 2.9 are for genuine borderline cases, not for papering over a
    systemic problem. If a real run comes back with a mismatch rate that
    high again, look at Phase 1's boundary-consistency plot for those
    sessions before assuming Function 2.9 is the right tool.

    Parameters
    ----------
    master_df : pandas.DataFrame
        From build_master_cell_table.

    Returns
    -------
    master_df : pandas.DataFrame
        Same table, with an added boolean 'layer_mismatch' column.
    mismatched_rows : pandas.DataFrame
        Just the rows where layer_mismatch is True, sorted by global_cell_id.
    """
    layer_cols = [c for c in master_df.columns if c.startswith('layer_')]

    master_df = master_df.copy()
    master_df['layer_mismatch'] = master_df[layer_cols].nunique(axis=1).gt(1)

    mismatched_rows = master_df[master_df['layer_mismatch']].sort_values('global_cell_id')

    n_mismatch = len(mismatched_rows)
    n_total = len(master_df)
    print(f"{n_mismatch}/{n_total} tracked cells have differing layer labels across sessions.")
    if n_mismatch > 0:
        print("\nMismatched cells:")
        print(mismatched_rows[['global_cell_id'] + layer_cols].to_string(index=False))

    return master_df, mismatched_rows


# ============================================================
# Function 2.9: interactive per-cell resolution of flagged mismatches
# ============================================================
def review_layer_mismatches_popup(tracking_result, mismatched_rows, crop_size=40,
                                   microns_per_pixel=MICRONS_PER_PIXEL):
    """
    Interactive per-cell resolution for cells flag_layer_mismatches flagged
    as disagreeing across sessions. Reuses plot_cell_across_group's proven
    footprint-crop visualization so you can visually judge the cell's
    actual appearance/position across sessions, with each session's
    already-computed layer label (from the master table) shown as text --
    deliberately not recomputing depth/boundary geometry here, since Phase
    1's saved curve is in different coordinate conventions depending on
    whether a session was 'propagated' (anchor frame) vs 'overridden' (that
    session's own raw frame), and the exact shift Phase 1 used isn't even
    persisted to disk -- re-deriving a boundary overlay in Phase 2 would
    risk a new, subtle frame-convention bug for the sake of a nicety.

    Controls
    --------
    '1'..'9'  -- trust that session's layer label as the consensus for this cell
    'x'       -- exclude this cell from layer-stratified analyses
    'b'/Left  -- back to the previous cell
    'q'       -- quit early (remaining cells stay unresolved)

    Parameters
    ----------
    tracking_result : dict
        From track_session_group.
    mismatched_rows : pandas.DataFrame
        From flag_layer_mismatches -- needs 'global_cell_id' and
        'layer_<label>' columns.
    crop_size : int
    microns_per_pixel : float

    Returns
    -------
    resolutions : dict
        {global_cell_id: {'action': 'trust_session' | 'excluded',
                           'trusted_label': str or None,
                           'consensus_layer': str or None}}
    """
    labels_order = tracking_result['labels_order']
    rows = mismatched_rows.reset_index(drop=True)
    total = len(rows)

    print(f"Reviewing {total} mismatched cells.")
    print("Controls: 'b'/Left = back, 'x' = exclude, 'q' = quit, "
          "or a number to trust that session:")
    for i, label in enumerate(labels_order, start=1):
        print(f"  {i}: {label}")
    print()

    resolutions = {}
    idx = 0

    while 0 <= idx < total:
        row = rows.iloc[idx]
        global_cell_id = int(row['global_cell_id'])

        fig = plot_cell_across_group(tracking_result, global_cell_id, crop_size=crop_size,
                                      microns_per_pixel=microns_per_pixel)

        layer_summary = "  |  ".join(f"{i}:{label}={row[f'layer_{label}']}"
                                      for i, label in enumerate(labels_order, start=1))
        status = resolutions.get(global_cell_id, {}).get('action', 'NOT REVIEWED')
        fig.suptitle(f"Mismatch review {idx+1}/{total} (cell {global_cell_id})  --  {status}\n"
                     f"{layer_summary}\n"
                     "Press a number to trust that session, 'x'=exclude, 'b'=back, 'q'=quit",
                     fontsize=11, fontweight='bold')

        action = {'value': None}
        valid_keys = {str(k): k - 1 for k in range(1, len(labels_order) + 1)}

        def on_key(event):
            if event.key in valid_keys:
                action['value'] = ('trust', valid_keys[event.key])
                fig.canvas.stop_event_loop()
            elif event.key == 'x':
                action['value'] = ('exclude', None)
                fig.canvas.stop_event_loop()
            elif event.key in ('b', 'left'):
                action['value'] = ('back', None)
                fig.canvas.stop_event_loop()
            elif event.key == 'q':
                action['value'] = ('quit', None)
                fig.canvas.stop_event_loop()

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show(block=False)

        while action['value'] is None and plt.fignum_exists(fig.number):
            fig.canvas.start_event_loop(0.1)

        if plt.fignum_exists(fig.number):
            plt.close(fig)

        if action['value'] is None:
            print("\nWindow closed without a decision -- quitting review.")
            break

        kind, payload = action['value']

        if kind == 'trust':
            trusted_label = labels_order[payload]
            consensus_layer = row[f'layer_{trusted_label}']
            resolutions[global_cell_id] = {
                'action': 'trust_session', 'trusted_label': trusted_label,
                'consensus_layer': consensus_layer
            }
            print(f"  Cell {global_cell_id}: trusting {trusted_label} -> {consensus_layer}")
            idx += 1
        elif kind == 'exclude':
            resolutions[global_cell_id] = {'action': 'excluded', 'trusted_label': None,
                                            'consensus_layer': None}
            print(f"  Cell {global_cell_id}: excluded")
            idx += 1
        elif kind == 'back':
            idx = max(0, idx - 1)
        elif kind == 'quit':
            print("\nQuitting review.")
            break

    n_resolved = len(resolutions)
    n_unresolved = total - n_resolved
    print(f"\n  Resolved: {n_resolved}")
    print(f"  Unresolved: {n_unresolved} (treat as excluded until reviewed)")

    return resolutions


# ============================================================
# Function 2.10: one-call driver for a single comparison group
# ============================================================
def track_and_build_table(session_catalog, group_name, session_labels, reference_label,
                           store_dir=None, max_footprint_pixels=500,
                           auto_accept_corr=0.7, auto_accept_dist=5.0,
                           microns_per_pixel=MICRONS_PER_PIXEL,
                           skip_manual_review=False, skip_mismatch_review=False):
    """
    One-call driver for a single comparison group: load -> track -> filter
    -> (manual review) -> save -> master table -> flag mismatches ->
    (mismatch review). Wraps Functions 2.2-2.9. This is what you actually
    run per group going forward, instead of chaining the individual
    functions by hand each time.

    Parameters
    ----------
    session_catalog : dict
        From discover_animal_sessions.
    group_name : str
    session_labels : list of str
        Must all be keys in session_catalog.
    reference_label : str
        Must be in session_labels.
    store_dir : str, optional
        Where to save the tracking group .h5 (defaults, inside
        save_tracking_group, to a TrackedGroups folder next to the
        reference session).
    max_footprint_pixels, auto_accept_corr, auto_accept_dist, microns_per_pixel :
        Passed through to filter_and_flag_matches.
    skip_manual_review : bool
        If True, skip the interactive manual_review_matches step --
        verified_matrix keeps only auto_accept cells (auto_reject and every
        manual_review cell excluded). Use for a quick/dry run.
    skip_mismatch_review : bool
        If True, skip review_layer_mismatches_popup -- mismatched cells stay
        flagged in master_df but unresolved.

    Returns
    -------
    result : dict with keys: tracking_result, filter_report, cell_categories,
        verified_matrix, decisions, group_save_path, master_df,
        mismatched_rows, resolutions, master_table_save_paths.
    """
    print("\n" + "=" * 90)
    print(f" TRACKING GROUP: {group_name}")
    print("=" * 90)

    sessions = {label: load_session_for_tracking(session_catalog[label]['plane0_path'])
                for label in session_labels}

    tracking_result = track_session_group(sessions, reference_label=reference_label,
                                           microns_per_pixel=microns_per_pixel)

    filtered_matrix, cell_categories, filter_report = filter_and_flag_matches(
        tracking_result, max_footprint_pixels=max_footprint_pixels,
        auto_accept_corr=auto_accept_corr, auto_accept_dist=auto_accept_dist,
        microns_per_pixel=microns_per_pixel
    )

    if skip_manual_review:
        verified_matrix = filtered_matrix.copy()
        for row, category in cell_categories.items():
            if category != 'auto_accept':
                verified_matrix[row, :] = -1
        decisions = {}
        print("skip_manual_review=True -- verified_matrix contains only auto_accept cells.")
    else:
        verified_matrix, decisions = manual_review_matches(
            tracking_result, cell_categories, microns_per_pixel=microns_per_pixel
        )

    plane0_paths = {label: session_catalog[label]['plane0_path'] for label in session_labels}
    group_save_path = save_tracking_group(
        tracking_result, verified_matrix, decisions, filter_report,
        plane0_paths, group_name, save_dir=store_dir
    )

    layer_curve_paths = {label: find_layer_curve_path(plane0_paths[label]) for label in session_labels}
    master_df = build_master_cell_table(tracking_result, verified_matrix, layer_curve_paths,
                                         microns_per_pixel=microns_per_pixel)

    master_df, mismatched_rows = flag_layer_mismatches(master_df)

    resolutions = {}
    if len(mismatched_rows) > 0 and not skip_mismatch_review:
        resolutions = review_layer_mismatches_popup(tracking_result, mismatched_rows,
                                                     microns_per_pixel=microns_per_pixel)
    elif len(mismatched_rows) > 0:
        print(f"skip_mismatch_review=True -- {len(mismatched_rows)} mismatched cells left unresolved.")

    # Persist master_df (+ mismatch resolutions) alongside save_tracking_group's
    # output -- see the fix note above save_master_cell_table for why this
    # wasn't happening before.
    master_table_save_paths = save_master_cell_table(
        master_df, group_name, save_dir=os.path.dirname(group_save_path), resolutions=resolutions
    )

    return {
        'tracking_result': tracking_result,
        'filter_report': filter_report,
        'cell_categories': cell_categories,
        'verified_matrix': verified_matrix,
        'decisions': decisions,
        'group_save_path': group_save_path,
        'master_df': master_df,
        'mismatched_rows': mismatched_rows,
        'resolutions': resolutions,
        'master_table_save_paths': master_table_save_paths,
    }


# ============================================================
# Entry point
# ============================================================
if __name__ == '__main__':

    ANIMAL_DIR = r"D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD"
    session_catalog = discover_animal_sessions(ANIMAL_DIR)

    # --------------------------------------------------------------------
    # Recommended real comparison groups (see project discussion):
    #   - Each DCZ day tracked as its OWN triplet (baseline anchor + that
    #     day's saline + that day's DCZ), NOT combined across days -- keeps
    #     N usable per comparison instead of collapsing across a long series.
    #   - Baseline anchor = the baseline day closest in time to the DCZ/
    #     openloop sessions (minimizes FOV drift for registration), e.g.
    #     Day5 if that's your animal's last baseline day before DREADD days.
    #   - Saline is included alongside DCZ in every triplet (not tracked
    #     separately) -- one tracking pass gives you baseline-vs-saline
    #     (sanity check), saline-vs-DCZ (the actual effect), and
    #     baseline-vs-DCZ, all for the same tracked-cell set.
    #   - OpenLoop active and stationary are two SEPARATE groups (each is
    #     its own within-day saline/DCZ pair) -- do not attempt to track
    #     cells between the active and stationary days themselves; that
    #     active-vs-stationary contrast is a between-group comparison of
    #     each group's own saline-vs-DCZ effect size, done at the analysis
    #     stage (Phase 5/6), not something that needs a cross-day cell link.
    #   - "Baseline1-5" (all 5 baseline days tracked together) is lower
    #     priority and arguably better answered via Track B (distributional,
    #     no tracking needed) unless within-cell baseline stability is
    #     specifically required -- a 5-way tracking group will have much
    #     lower N than these 2-3-way groups.
    #
    # Fill in real session labels from session_catalog's printed output
    # (or use pick_tracking_group_interactively(session_catalog) to build
    # entries via the checkbox popup instead of typing labels by hand).
    # --------------------------------------------------------------------
    # First pass: saline-vs-dcz only, no baseline anchor -- keeps the
    # tracking pass to its minimum useful scope (just what Track A's
    # actual comparison needs) and gives the largest possible tracked-N
    # per pair, since a 2-way match will always find at least as many
    # cells as the corresponding 3-way match with Day5 added. Baseline
    # comparisons (baseline-vs-saline, baseline-vs-dcz) can be added back
    # as a later pass by re-adding 'Day5' to session_labels if needed --
    # group_name kept as 'DCZ1'/'DCZ2'/'DCZ3' to match Phase 4/5/6's Track
    # B comparison-group names, so Track A's later joins line up by name.
    # session_labels use the REAL discover_animal_sessions labels
    # ({parent_folder_name}_SALINE / _DCZ, same logic as Phase 4/6's
    # discover_smi_sessions) -- confirmed against this animal's actual
    # folder names from Phase 6's session discovery earlier this project,
    # not the DCZ1/DCZ2/DCZ3 shorthand we use conversationally. Verify
    # these against this run's own printed session_catalog before
    # trusting them -- if discover_animal_sessions hit a label collision
    # here that Phase 4/6 didn't (different discovery method, plane0 vs
    # smi_results_dreadd.h5), the real label could carry a
    # '__{tseries_name}' disambiguation suffix instead.
    TRACKING_GROUPS = [
        {
            'group_name': 'DCZ1',
            'session_labels': ['260724_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_1_SALINE',
                               '260724_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_1_DCZ'],
            'reference_label': '260724_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_1_SALINE',
        },
        {
            'group_name': 'DCZ2',
            'session_labels': ['260726_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_2_SALINE',
                               '260726_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_2_DCZ'],
            'reference_label': '260726_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_2_SALINE',
        },
        {
            'group_name': 'DCZ3',
            'session_labels': ['260728_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_3_SALINE',
                               '260728_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_3_DCZ'],
            'reference_label': '260728_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_3_SALINE',
        },
        # NOTE: these two are still shorthand placeholders, not real
        # discover_animal_sessions labels -- check this run's printed
        # session_catalog for the real {parent_folder_name}_SALINE/_DCZ
        # labels before uncommenting (same fix just applied to DCZ1/2/3).
        # {
        #     'group_name': 'Active_OL',
        #     'session_labels': ['OpenLoopActive_SALINE', 'OpenLoopActive_DCZ'],
        #     'reference_label': 'OpenLoopActive_SALINE',
        # },
        # {
        #     'group_name': 'Stationary_OL',
        #     'session_labels': ['OpenLoopStationary_SALINE', 'OpenLoopStationary_DCZ'],
        #     'reference_label': 'OpenLoopStationary_SALINE',
        # },
        # ... add/edit entries to match your real session_catalog labels,
        # or comment all of these out and use:
        #   labels, ref = pick_tracking_group_interactively(session_catalog)
        #   TRACKING_GROUPS = [{'group_name': 'MyGroup', 'session_labels': labels,
        #                       'reference_label': ref}]
    ]

    group_results = {}
    for cfg in TRACKING_GROUPS:
        group_results[cfg['group_name']] = track_and_build_table(
            session_catalog, cfg['group_name'], cfg['session_labels'], cfg['reference_label']
        )
