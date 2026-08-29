"""
1.LayerAssignment_Curve.py
Phase 1 (DREADD saline/DCZ cohort) -- FOV Correction & Layer Assignment.

Default approach: chained registration across an animal's sessions -> one
denoised, enhanced averaged reference image -> ONE shared L4-band click
(with an immediate vertical-nudge adjustment available, since eyeballing
exact position while clicking is hard) -> propagate that curve to every
session via the same registration shift, with a per-session
confirm/nudge-adjust/full-reclick-override safety net.

History: an earlier version of this script clicked the L4 band center
independently on every session. That approach was measured in real Phase 2
testing to produce a ~66% tracked-cell layer-mismatch rate across just a
3-session group (Day1 vs a saline/DCZ pair). Most mismatches were between
*adjacent* layers (L5<->L6, L4<->L5, etc.) -- the signature of small
boundary-position differences between independent clicks tipping borderline
cells over a threshold, not real biological drift. Since the same
per-session independent click also feeds Track B (all-cells,
no-tracking-required analyses), this wasn't just a Phase-2/tracking problem
-- it would have added noise to every session-level layer comparison in
this project. (A second, larger contributor to that investigation's
apparent mismatch rate turned out to be a separate indexing bug in Phase
2's own join logic -- 2.CellTracking.ipynb's _load_layer_of_cell was keying
its lookup by raw suite2p ROI index instead of by iscell-filtered position,
fixed there directly. But the click-consistency problem this script's
design solves was real too, and both fixes were needed to get the
mismatch rate down to a small number of genuinely borderline cells.)

Pipeline (per animal):
  1. Sort all of the animal's sessions into true chronological order, using
     the acquisition date+time embedded in each TSeries folder's own name
     (consistent across every layout this project uses -- baseline Day<N>
     folders and flat SALINE/DCZ pairs alike).
  2. Register each session to its immediate TEMPORAL NEIGHBOR (not directly
     to session 1) via phase_cross_correlation, and compose those pairwise
     shifts into each session's cumulative shift relative to session 1's
     frame. Chaining through small, reliable day-to-day shifts avoids ever
     needing one fragile long-range registration between sessions that are
     weeks apart and may have drifted a lot (observed in this cohort).
  3. Shift every session's padded max-projection into that common frame,
     average them, then sharpen + CLAHE-enhance the composite -- a single
     denoised, crisper reference (this image is only ever used for
     display/clicking, never quantitatively, so enhancing it doesn't affect
     anything downstream).
  4. Click the L4 band ONCE on that composite (vertical-nudge adjustment
     available immediately if the click isn't quite right) -- one
     curve_fn/boundary_offsets, in session 1's coordinate frame.
  5. For every session, evaluate that SAME shared curve at that session's
     own cells' positions, translated into the common frame via its own
     chained shift (identify_layers_from_shifted_curve). Show a per-session
     confirm popup (review_propagated_layers_popup, side-by-side against
     the original averaged-reference click); if rejected, open a
     vertical-only adjustment window (adjust_layer_curve_popup) that keeps
     the shared slope fixed and lets you nudge for genuine per-session
     depth (Z) drift -- only falling back to a full independent re-click
     (pick_and_confirm_layers) if the slope itself, not just the depth,
     genuinely needs to change.
  6. Save each session's result (save_layer_curve_results), tagged with a
     'source' attribute ('propagated' | 'propagated_adjusted' |
     'overridden') and, for adjusted sessions, the exact vertical_nudge_px
     used -- so it's traceable later which method produced a given
     session's boundary.

Output: *_layer_curve_results_averaged.h5, saved directly in that session's
TSeries folder (same location/discovery convention as preproc.h5 and
*_smi_results.h5), indexed by the iscell-filtered ROI order -- so Phase 2
(tracking) and Phase 3 (SMI calc) can join against it without any index
translation. Kept the "_averaged" suffix deliberately even now that this is
the only Phase 1 script, rather than renaming to the plain
*_layer_curve_results.h5 the old independent-click version used -- that
would silently overwrite the real independent-click results already
computed for all 15 sessions across both animals. Phase 2's
find_layer_curve_path already prefers this suffix.

Developed incrementally in 1.LayerAssignment.ipynb (same folder) -- start
there for any further changes, not directly in this file.

JSY, 2026
"""

import os
import re
import datetime
import numpy as np
import h5py
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # interactive ginput() clicking + popup windows
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.ndimage import shift as ndi_shift, gaussian_filter
from skimage.registration import phase_cross_correlation
from skimage import exposure

rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20

_LAYER_COLORS = {
    'L2/3': '#1f77b4',  # Blue  -- matches SpatialModulationIndexLayerSpecific's convention
    'L4': '#ff7f0e',    # Orange
    'L5': '#2ca02c',    # Green
    'L6': '#d62728',    # Red
}


# ============================================================
# Core click / fit / assign / review / persist functions
# ============================================================
def get_fullframe_max_proj(ops):
    """
    Embed suite2p's cropped ops['max_proj'] into a full (Ly, Lx) canvas at
    its registered yrange/xrange offset, so it aligns pixel-for-pixel with
    med_coords and every other full-frame image used elsewhere.
    """
    Ly, Lx = ops['Ly'], ops['Lx']
    y0, y1 = ops['yrange']
    x0, x1 = ops['xrange']
    max_proj = ops['max_proj']

    expected_shape = (y1 - y0, x1 - x0)
    if max_proj.shape != expected_shape:
        raise ValueError(
            f"max_proj shape {max_proj.shape} doesn't match yrange/xrange "
            f"{expected_shape} -- check this session's ops.npy."
        )

    fullframe = np.zeros((Ly, Lx), dtype=max_proj.dtype)
    fullframe[y0:y1, x0:x1] = max_proj
    return fullframe


def click_boundary_points(fullframe_img, med_coords,
                           title='Click points along the L4 band center, left to right.\n'
                                 'Right-click = undo last point. Enter = done (need >= 2 points).',
                           reference_curve_fn=None, reference_shift=(0.0, 0.0)):
    """
    Interactively click points tracing the L4 band center.

    reference_curve_fn / reference_shift : optional
        If given, draws a faint dashed reference line on the image before
        you click -- typically the propagated curve you just rejected,
        transformed into this image's own raw frame via reference_shift
        (same convention as review_propagated_layers_popup:
        y_raw(x) = reference_curve_fn(x + dx) - dy) -- so you have visual
        context for how far off it looked, instead of clicking on a blank
        image with no reference at all.
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(fullframe_img, cmap='gray',
              vmin=np.percentile(fullframe_img, 2), vmax=np.percentile(fullframe_img, 98))
    ax.scatter(med_coords[:, 1], med_coords[:, 0], s=8, alpha=0.35, c='cyan')

    if reference_curve_fn is not None:
        dy, dx = reference_shift
        x_line = np.linspace(0, fullframe_img.shape[1], 200)
        ax.plot(x_line, reference_curve_fn(x_line + dx) - dy, '--', color='orange',
                linewidth=1.5, alpha=0.85, label='rejected propagated line (for reference)')
        ax.legend(loc='upper right', fontsize=9)

    ax.set_title(title, fontsize=13)
    plt.tight_layout()

    pts = plt.ginput(n=-1, timeout=0)
    plt.close(fig)

    if len(pts) < 2:
        raise ValueError(f"Need at least 2 points to fit a boundary line, got {len(pts)}.")

    points = np.array(sorted(pts, key=lambda p: p[0]))

    print(f"Clicked {len(points)} points (sorted left-to-right):")
    for x, y in points:
        print(f"  x={x:.1f}, y={y:.1f}")

    return points


def fit_boundary_curve(points, degree=1):
    """Fit a polynomial y = f(x) through manually-clicked boundary points."""
    n_points = len(points)
    if n_points < degree + 1:
        raise ValueError(
            f"degree={degree} needs at least {degree + 1} points, "
            f"but only {n_points} were given. Re-click more points or lower degree."
        )

    x = points[:, 0]
    y = points[:, 1]

    coeffs = np.polyfit(x, y, degree)
    curve_fn = np.poly1d(coeffs)

    fitted_y = curve_fn(x)
    residuals = y - fitted_y
    rms = np.sqrt(np.mean(residuals ** 2))

    print(f"Fitted degree-{degree} curve through {n_points} points:")
    print(f"  {curve_fn}")
    print(f"  Residuals (clicked y - fit y): min={residuals.min():+.2f}px, "
          f"max={residuals.max():+.2f}px, RMS={rms:.2f}px")

    return coeffs, curve_fn


def compute_relative_depth(med_coords, curve_fn):
    """Vertical offset of each cell from the fitted boundary curve."""
    y = med_coords[:, 0]
    x = med_coords[:, 1]
    return y - curve_fn(x)


def identify_layers_from_curve(med_coords, curve_fn, um_per_pixel,
                                layer4_half_width_um=70, layer5_offset_um=150):
    """Assign each cell to a cortical layer based on offset from curve_fn (no shift)."""
    depth_rel = compute_relative_depth(med_coords, curve_fn)

    layer4_half_width_px = layer4_half_width_um / um_per_pixel
    layer5_offset_px = layer5_offset_um / um_per_pixel

    l4_upper = -layer4_half_width_px
    l4_lower = layer4_half_width_px
    l5_lower = l4_lower + layer5_offset_px

    layer23_cells = np.where(depth_rel < l4_upper)[0]
    layer4_cells = np.where((depth_rel >= l4_upper) & (depth_rel < l4_lower))[0]
    layer5_cells = np.where((depth_rel >= l4_lower) & (depth_rel < l5_lower))[0]
    layer6_cells = np.where(depth_rel >= l5_lower)[0]

    print(f"L2/3: {len(layer23_cells)} cells (depth_rel < {l4_upper:.1f}px)")
    print(f"L4:   {len(layer4_cells)} cells ({l4_upper:.1f} <= depth_rel < {l4_lower:.1f}px)")
    print(f"L5:   {len(layer5_cells)} cells ({l4_lower:.1f} <= depth_rel < {l5_lower:.1f}px)")
    print(f"L6:   {len(layer6_cells)} cells (depth_rel >= {l5_lower:.1f}px)")

    layer_cells = {'L2/3': layer23_cells, 'L4': layer4_cells, 'L5': layer5_cells, 'L6': layer6_cells}
    boundary_offsets = {'L4_upper': l4_upper, 'L4_lower': l4_lower, 'L5_lower': l5_lower}
    return layer_cells, boundary_offsets


def review_layer_assignment_popup(fullframe_img, med_coords, layer_cells,
                                   curve_fn, boundary_offsets, points):
    """Two-panel popup: plain FOV | FOV + layer-colored ROIs + boundary curves. y=accept, n=reject."""
    Ly, Lx = fullframe_img.shape
    vmin, vmax = np.percentile(fullframe_img, 2), np.percentile(fullframe_img, 98)

    fig, (ax_plain, ax_layers) = plt.subplots(1, 2, figsize=(16, 8))
    try:
        fig.canvas.manager.set_window_title('LAYER REVIEW -- press y/n')
    except Exception:
        pass

    ax_plain.imshow(fullframe_img, cmap='gray', vmin=vmin, vmax=vmax)
    ax_plain.set_title('FOV only (max-projection)')
    ax_plain.axis('off')

    ax_layers.imshow(fullframe_img, cmap='gray', vmin=vmin, vmax=vmax)
    for layer_name, cell_idx in layer_cells.items():
        if len(cell_idx) == 0:
            continue
        color = _LAYER_COLORS.get(layer_name, 'white')
        ax_layers.scatter(med_coords[cell_idx, 1], med_coords[cell_idx, 0],
                           s=14, alpha=0.85, color=color, label=f'{layer_name} ({len(cell_idx)})')

    x_line = np.linspace(0, Lx, 200)
    ax_layers.plot(x_line, curve_fn(x_line), '--', color='yellow', linewidth=1.5,
                    label='L4 center (clicked)')
    ax_layers.plot(x_line, curve_fn(x_line) + boundary_offsets['L4_upper'], '--',
                    color='white', linewidth=1.2, label='L2/3-L4')
    ax_layers.plot(x_line, curve_fn(x_line) + boundary_offsets['L4_lower'], '--',
                    color='white', linewidth=1.2, label='L4-L5')
    ax_layers.plot(x_line, curve_fn(x_line) + boundary_offsets['L5_lower'], '--',
                    color='white', linewidth=1.2, label='L5-L6')
    ax_layers.scatter(points[:, 0], points[:, 1], s=60, c='red', marker='x',
                       label='clicked points', zorder=10)

    ax_layers.set_xlim(0, Lx)
    ax_layers.set_ylim(Ly, 0)
    ax_layers.set_title('FOV + layer assignment')
    ax_layers.axis('off')
    ax_layers.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)

    decision = {'accepted': None}

    def on_key(event):
        if event.key == 'y':
            decision['accepted'] = True
            fig.canvas.stop_event_loop()
        elif event.key == 'n':
            decision['accepted'] = False
            fig.canvas.stop_event_loop()

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.suptitle("Click this figure to focus it, then press 'y' to ACCEPT or 'n' to REJECT",
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.show(block=False)

    while decision['accepted'] is None and plt.fignum_exists(fig.number):
        fig.canvas.start_event_loop(0.1)

    if plt.fignum_exists(fig.number):
        plt.close(fig)

    if decision['accepted'] is None:
        print("Window closed without pressing y/n -- treating as reject.")
        return False

    return decision['accepted']


def pick_and_confirm_layers(fullframe_img, med_coords, um_per_pixel, degree=1,
                             reference_curve_fn=None, reference_shift=(0.0, 0.0)):
    """
    Click -> fit -> assign -> review -> (if rejected) vertical-adjustment
    window -> (if still not right) re-click from scratch, looping until
    accepted. Used both for the ONE shared click on the averaged reference,
    and as the per-session manual-override fallback when a propagated
    boundary is rejected.

    The vertical-adjustment step (adjust_layer_curve_popup, called with
    shift=(0.0, 0.0) since this is always a fresh independent click, not a
    propagated one) exists because eyeballing the exact right position
    while placing the original points is genuinely hard -- nudging the
    result afterward is much faster than repeatedly re-clicking fresh
    points to get it right. The curve's slope is never touched by nudging,
    only its vertical position, folded back into a standalone curve_fn/
    coeffs before returning (see the coefficient-shift math below) so
    everything downstream sees an ordinary fitted curve either way.

    reference_curve_fn / reference_shift : optional
        Passed straight to click_boundary_points on every attempt in this
        loop -- draws a faint reference line (e.g. the just-rejected
        propagated curve) so re-clicking isn't done on a blank image with
        no context for how far off the propagation looked.
    """
    attempt = 0
    while True:
        attempt += 1
        print(f"\n{'='*60}\nAttempt {attempt}\n{'='*60}")

        points = click_boundary_points(fullframe_img, med_coords,
                                        reference_curve_fn=reference_curve_fn,
                                        reference_shift=reference_shift)
        coeffs, curve_fn = fit_boundary_curve(points, degree=degree)
        layer_cells, boundary_offsets = identify_layers_from_curve(
            med_coords, curve_fn, um_per_pixel=um_per_pixel
        )

        accepted = review_layer_assignment_popup(
            fullframe_img, med_coords, layer_cells, curve_fn, boundary_offsets, points
        )

        if accepted:
            print(f"Accepted on attempt {attempt}.")
            return points, coeffs, curve_fn, layer_cells, boundary_offsets

        print("Rejected -- opening vertical-adjustment window "
              "(press 'f' there to re-click from scratch instead).")
        outcome, adjusted_shift, adjusted_layer_cells, adjusted_boundary_offsets = adjust_layer_curve_popup(
            fullframe_img, med_coords, curve_fn, boundary_offsets, (0.0, 0.0), um_per_pixel,
            session_label=f'attempt {attempt}'
        )

        if outcome == 'adjusted':
            nudge_dy = adjusted_shift[0]
            # Fold the nudge into a standalone curve: adjusted(x) = curve_fn(x) - nudge_dy
            # is exactly what the popup displayed/evaluated with shift=(0,0), so this
            # reproduces it as an ordinary poly1d with no separate shift to carry around.
            adjusted_coeffs = coeffs.copy()
            adjusted_coeffs[-1] -= nudge_dy
            adjusted_curve_fn = np.poly1d(adjusted_coeffs)
            print(f"Accepted after vertical adjustment on attempt {attempt} "
                  f"(nudge={nudge_dy:+.1f}px, {nudge_dy * um_per_pixel:+.1f}um).")
            return points, adjusted_coeffs, adjusted_curve_fn, adjusted_layer_cells, adjusted_boundary_offsets

        print("Re-clicking from scratch.")


def layer_cells_to_labels(layer_cells, n_cells, layer_names=('L2/3', 'L4', 'L5', 'L6')):
    """Flatten a {layer_name: indices} dict into a per-cell integer label array."""
    layer_codes = np.full(n_cells, -1, dtype=np.int8)
    for code, name in enumerate(layer_names):
        if name in layer_cells:
            layer_codes[layer_cells[name]] = code

    n_unassigned = np.sum(layer_codes == -1)
    if n_unassigned > 0:
        print(f"WARNING: {n_unassigned}/{n_cells} cells have no layer code "
              f"(layer_cells covered names other than {layer_names}?).")

    return layer_codes, layer_names


def save_layer_curve_results(save_path, session_label, cell_idx, layer_codes, layer_names,
                              points, coeffs, degree, um_per_pixel, boundary_offsets,
                              source='independent', extra_attrs=None):
    """
    Save one session's Phase 1 output to HDF5. Same format as the original
    file, plus a 'source' attribute ('propagated' | 'propagated_adjusted' |
    'overridden' | 'independent') recording which workflow actually
    produced this session's boundary, and an optional extra_attrs dict for
    additional provenance (e.g. {'vertical_nudge_px': ...} for
    'propagated_adjusted' sessions).
    """
    with h5py.File(save_path, 'w') as f:
        f.attrs['session_label'] = session_label
        f.attrs['degree'] = degree
        f.attrs['um_per_pixel'] = um_per_pixel
        f.attrs['source'] = source
        for key, val in boundary_offsets.items():
            f.attrs[f'boundary_offset_{key}'] = val
        if extra_attrs:
            for key, val in extra_attrs.items():
                f.attrs[key] = val

        f.create_dataset('cell_idx', data=cell_idx)
        f.create_dataset('layer_codes', data=layer_codes)
        f.create_dataset('layer_names', data=np.array(layer_names, dtype='S10'))
        f.create_dataset('points', data=points)
        f.create_dataset('coeffs', data=coeffs)

    print(f"Saved layer curve results for '{session_label}' (source={source}) -> {save_path}")


def load_layer_curve_results(save_path):
    """Load one session's Phase 1 output back from HDF5, including 'source'."""
    with h5py.File(save_path, 'r') as f:
        session_label = f.attrs['session_label']
        degree = int(f.attrs['degree'])
        um_per_pixel = float(f.attrs['um_per_pixel'])
        source = str(f.attrs['source']) if 'source' in f.attrs else 'independent'
        boundary_offsets = {
            'L4_upper': float(f.attrs['boundary_offset_L4_upper']),
            'L4_lower': float(f.attrs['boundary_offset_L4_lower']),
            'L5_lower': float(f.attrs['boundary_offset_L5_lower']),
        }
        cell_idx = f['cell_idx'][:]
        layer_codes = f['layer_codes'][:]
        layer_names = tuple(n.decode() if isinstance(n, bytes) else n
                             for n in f['layer_names'][:])
        points = f['points'][:]
        coeffs = f['coeffs'][:]

    curve_fn = np.poly1d(coeffs)
    layer_of_cell = {
        int(roi_idx): layer_names[code]
        for roi_idx, code in zip(cell_idx, layer_codes) if code >= 0
    }

    return {
        'session_label': session_label,
        'degree': degree,
        'um_per_pixel': um_per_pixel,
        'source': source,
        'boundary_offsets': boundary_offsets,
        'cell_idx': cell_idx,
        'layer_codes': layer_codes,
        'layer_names': layer_names,
        'points': points,
        'coeffs': coeffs,
        'curve_fn': curve_fn,
        'layer_of_cell': layer_of_cell,
    }


def plot_curve_boundary_consistency(session_labels, fullframe_imgs, curve_results,
                                     reference_label=None, save_path=None):
    """Overlay each session's curve on one reference FOV, aligned by translation only."""
    available = [s for s in session_labels if s in curve_results]
    if len(available) == 0:
        raise ValueError("None of the given session_labels have entries in curve_results")

    if reference_label is None:
        reference_label = available[0]
    if reference_label not in available:
        raise ValueError(f"reference_label '{reference_label}' has no entry in curve_results")

    label_to_img = dict(zip(session_labels, fullframe_imgs))
    ref_img = label_to_img[reference_label]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(ref_img, cmap='gray',
              vmin=np.percentile(ref_img, 2), vmax=np.percentile(ref_img, 98))

    colors = plt.cm.hsv(np.linspace(0, 0.85, len(available)))

    for color, label in zip(colors, available):
        img = label_to_img[label]
        curve_fn = curve_results[label]['curve_fn']
        source = curve_results[label].get('source', '?')

        if label == reference_label:
            dy, dx = 0.0, 0.0
        else:
            shift_yx, _, _ = phase_cross_correlation(ref_img, img, upsample_factor=10)
            dy, dx = shift_yx[0], shift_yx[1]

        x_line = np.linspace(0, img.shape[1], 200)
        y_line = curve_fn(x_line)
        ax.plot(x_line + dx, y_line + dy, '-', color=color, linewidth=2,
                 label=f"{label} ({source})")

    ax.set_title(f"L4 boundary curve consistency across sessions\n(reference: {reference_label})")
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Boundary consistency figure saved to {save_path}")

    return fig


def summarize_layer_curve_consistency(session_labels, curve_results):
    """Per-session summary table: curve slope/intercept, source, per-layer counts/percentages."""
    rows = []
    for label in session_labels:
        r = curve_results[label]
        layer_codes = r['layer_codes']
        layer_names = r['layer_names']
        n_cells = len(layer_codes)

        row = {
            'session_label': label,
            'source': r.get('source', '?'),
            'n_cells': n_cells,
            'degree': r['degree'],
            'slope': r['coeffs'][0] if r['degree'] == 1 else np.nan,
            'intercept': r['coeffs'][-1],
            'n_clicked_points': len(r['points']),
        }
        for code, name in enumerate(layer_names):
            count = int(np.sum(layer_codes == code))
            row[f'{name}_n'] = count
            row[f'{name}_pct'] = 100 * count / n_cells if n_cells else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    with pd.option_context('display.width', 140, 'display.max_columns', None):
        print(df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

    return df


# ============================================================
# NEW: chronological ordering
# ============================================================
def _extract_session_datetime(session_dir):
    """
    Parse the acquisition date+time from a TSeries folder's own name, e.g.
    'TSeries-07192026-0941-001' -> July 19 2026, 09:41. This MMDDYYYY-HHMM
    prefix format is consistent across every session layout in this project
    (baseline Day<N> folders and flat SALINE/DCZ pairs alike), so it's a
    reliable chronological sort key regardless of folder-naming convention.

    Parameters
    ----------
    session_dir : str
        Path to a TSeries folder.

    Returns
    -------
    dt : datetime.datetime
    """
    name = os.path.basename(str(session_dir))
    m = re.search(r'TSeries-(\d{2})(\d{2})(\d{4})-(\d{4})', name)
    if not m:
        raise ValueError(
            f"Could not parse acquisition date/time from TSeries folder name: '{name}'. "
            "Expected the standard 'TSeries-MMDDYYYY-HHMM...' prefix."
        )
    month, day, year, hhmm = m.groups()
    hour, minute = int(hhmm[:2]), int(hhmm[2:])
    return datetime.datetime(int(year), int(month), int(day), hour, minute)


def sort_sessions_chronologically(session_dirs):
    """
    Sort session TSeries-folder paths by actual acquisition date+time, so
    chained registration always walks sessions in true temporal order
    regardless of the order they were listed in a config.

    Parameters
    ----------
    session_dirs : list of str

    Returns
    -------
    list of str, sorted by _extract_session_datetime.
    """
    return sorted(session_dirs, key=_extract_session_datetime)


# ============================================================
# NEW: chained registration + averaged reference
# ============================================================
def compute_chained_shifts(fullframe_imgs_ordered):
    """
    Register each session to its immediate TEMPORAL NEIGHBOR (not directly
    to the first session), and compose those pairwise shifts into each
    session's cumulative shift relative to the first session's frame.

    Chaining through consecutive, visually-similar sessions is far more
    robust than one long-range registration between sessions that are weeks
    apart and may have drifted a lot -- each link only has to handle a
    small, reliable day-to-day shift.

    Parameters
    ----------
    fullframe_imgs_ordered : dict or list of (label, img) tuples
        MUST already be in true chronological order (see
        sort_sessions_chronologically) -- shifts are only correct if
        consecutive entries are actually temporal neighbors. A plain dict
        works since Python dicts preserve insertion order.

    Returns
    -------
    shifts : dict
        {label: (dy, dx)} in pixels, cumulative relative to the first
        session (whose own shift is (0.0, 0.0)).
    """
    items = list(fullframe_imgs_ordered.items()) if isinstance(fullframe_imgs_ordered, dict) \
        else list(fullframe_imgs_ordered)

    shifts = {}
    cumulative = np.array([0.0, 0.0])
    prev_img = None

    for i, (label, img) in enumerate(items):
        if i == 0:
            shifts[label] = (0.0, 0.0)
            print(f"  {label}: anchor (0, 0)")
        else:
            shift_yx, _, _ = phase_cross_correlation(prev_img, img, upsample_factor=10)
            cumulative = cumulative + np.array(shift_yx)
            shifts[label] = (float(cumulative[0]), float(cumulative[1]))
            print(f"  {label}: pairwise shift from previous session = "
                  f"({shift_yx[0]:+.2f}, {shift_yx[1]:+.2f}), "
                  f"cumulative = ({cumulative[0]:+.2f}, {cumulative[1]:+.2f})")
        prev_img = img

    return shifts


def build_averaged_reference(fullframe_imgs_ordered, shifts, sharpen=True,
                              sharpen_sigma=2.0, sharpen_amount=1.0, use_clahe=True):
    """
    Shift every session's padded max-projection into the common (first-
    session) frame using compute_chained_shifts's output, then average them
    into one denoised composite reference image.

    Averaging N images inherently blurs out anything that isn't
    pixel-perfectly aligned across sessions -- small residual registration
    error, or genuine session-to-session appearance differences -- on top
    of whatever the individual images' own blur already was. Two optional
    post-processing passes counteract that for the purpose of clicking a
    confident boundary (this image is never used quantitatively elsewhere
    in the pipeline, only for display/clicking, so enhancing it doesn't
    affect anything else):

    sharpen, sharpen_sigma, sharpen_amount : bool, float, float
        Unsharp-mask sharpening: averaged += amount * (averaged -
        gaussian_blur(averaged, sigma)). Boosts edge contrast (crisper
        band boundary) without re-introducing raw per-session noise, since
        it operates on the already-averaged image.
    use_clahe : bool
        Also apply contrast-limited adaptive histogram equalization
        (skimage.exposure.equalize_adapthist) after sharpening -- boosts
        LOCAL contrast, which the plain global percentile-based display
        stretch used elsewhere in this pipeline can't do, and tends to
        help microscopy images where brightness varies a lot across the FOV.

    Parameters
    ----------
    fullframe_imgs_ordered : dict or list of (label, img)
        Same sessions as `shifts`.
    shifts : dict
        {label: (dy, dx)}, from compute_chained_shifts.

    Returns
    -------
    averaged_image : numpy.ndarray, shape (Ly, Lx)
    """
    items = list(fullframe_imgs_ordered.items()) if isinstance(fullframe_imgs_ordered, dict) \
        else list(fullframe_imgs_ordered)

    shifted_stack = []
    for label, img in items:
        dy, dx = shifts[label]
        shifted = ndi_shift(img, shift=(dy, dx), mode='constant', cval=0)
        shifted_stack.append(shifted)

    averaged_image = np.mean(np.stack(shifted_stack, axis=0), axis=0)

    if sharpen:
        blurred = gaussian_filter(averaged_image, sigma=sharpen_sigma)
        averaged_image = averaged_image + sharpen_amount * (averaged_image - blurred)
        averaged_image = np.clip(averaged_image, 0, None)

    if use_clahe:
        img_range = averaged_image.max() - averaged_image.min()
        if img_range > 0:
            normalized = (averaged_image - averaged_image.min()) / img_range
            averaged_image = exposure.equalize_adapthist(normalized, clip_limit=0.01)
        else:
            print("  WARNING: averaged reference has zero dynamic range -- skipping CLAHE.")

    tags = ('sharpened' if sharpen else '', 'CLAHE' if use_clahe else '')
    tag_str = f" ({', '.join(t for t in tags if t)})" if any(tags) else ''
    print(f"Built averaged reference from {len(items)} sessions{tag_str}.")
    return averaged_image


# ============================================================
# NEW: propagate the shared curve to one session
# ============================================================
def identify_layers_from_shifted_curve(med_coords, curve_fn, shift, um_per_pixel,
                                        layer4_half_width_um=70, layer5_offset_um=150,
                                        verbose=True):
    """
    Like identify_layers_from_curve, but translates med_coords into the
    shared curve's coordinate frame via this session's own chained shift
    before evaluating: depth_rel = (y_raw + dy) - curve_fn(x_raw + dx).

    Parameters
    ----------
    med_coords : numpy.ndarray, shape (n_cells, 2)
        (y, x) per cell, in THIS session's own raw pixel frame.
    curve_fn : callable
        The ONE shared curve, fit in the anchor session's frame.
    shift : tuple of float
        (dy, dx) for this session, from compute_chained_shifts.
    um_per_pixel, layer4_half_width_um, layer5_offset_um : float
    verbose : bool
        Set False when calling this repeatedly in a tight loop (e.g. every
        keypress during adjust_layer_curve_popup's live nudge preview)
        to avoid spamming the console.

    Returns
    -------
    layer_cells, boundary_offsets
        Same shape as identify_layers_from_curve's output.
    """
    dy, dx = shift
    y = med_coords[:, 0] + dy
    x = med_coords[:, 1] + dx
    depth_rel = y - curve_fn(x)

    layer4_half_width_px = layer4_half_width_um / um_per_pixel
    layer5_offset_px = layer5_offset_um / um_per_pixel

    l4_upper = -layer4_half_width_px
    l4_lower = layer4_half_width_px
    l5_lower = l4_lower + layer5_offset_px

    layer23_cells = np.where(depth_rel < l4_upper)[0]
    layer4_cells = np.where((depth_rel >= l4_upper) & (depth_rel < l4_lower))[0]
    layer5_cells = np.where((depth_rel >= l4_lower) & (depth_rel < l5_lower))[0]
    layer6_cells = np.where(depth_rel >= l5_lower)[0]

    if verbose:
        print(f"L2/3: {len(layer23_cells)} cells (depth_rel < {l4_upper:.1f}px)")
        print(f"L4:   {len(layer4_cells)} cells ({l4_upper:.1f} <= depth_rel < {l4_lower:.1f}px)")
        print(f"L5:   {len(layer5_cells)} cells ({l4_lower:.1f} <= depth_rel < {l5_lower:.1f}px)")
        print(f"L6:   {len(layer6_cells)} cells (depth_rel >= {l5_lower:.1f}px)")

    layer_cells = {'L2/3': layer23_cells, 'L4': layer4_cells, 'L5': layer5_cells, 'L6': layer6_cells}
    boundary_offsets = {'L4_upper': l4_upper, 'L4_lower': l4_lower, 'L5_lower': l5_lower}
    return layer_cells, boundary_offsets


def review_propagated_layers_popup(fullframe_img, med_coords, layer_cells, curve_fn,
                                    boundary_offsets, shift, shared_points, session_label,
                                    averaged_image, med_coords_anchor, um_per_pixel):
    """
    Two-panel confirm popup for a PROPAGATED layer assignment:
      - left: the AVERAGED reference image where the one shared curve was
        originally clicked, with that curve/boundary lines and the anchor
        session's own cells colored by layer -- so you always have the
        original click to refer back to, not just this session's
        transformed view in isolation
      - right: this session's FOV + propagated layer-colored cells +
        boundary curves, transformed back into its own raw pixel frame
        (inverting the shift used to evaluate depth_rel:
        y_raw(x_raw) = curve_fn(x_raw + dx) - dy), plus the shared click
        points transformed the same way
    ROI markers are kept small (s=4) in both panels so they don't obscure
    the actual FOV underneath them.
    'y' accepts, 'n' rejects (caller should fall back to a manual re-click
    for just this session).

    Parameters
    ----------
    fullframe_img : numpy.ndarray, shape (Ly, Lx)
        This session's own padded max-projection (unshifted).
    med_coords : numpy.ndarray, shape (n_cells, 2)
    layer_cells : dict
        From identify_layers_from_shifted_curve, for THIS session.
    curve_fn : callable
        The shared curve (anchor frame).
    boundary_offsets : dict
    shift : tuple of float
        (dy, dx) for this session.
    shared_points : numpy.ndarray, shape (n_points, 2)
        The ONE set of clicked points (anchor frame).
    session_label : str
    averaged_image : numpy.ndarray, shape (Ly, Lx)
        The averaged reference image the shared curve was clicked on.
    med_coords_anchor : numpy.ndarray, shape (n_anchor_cells, 2)
        Anchor session's own cells, native averaged-image frame.
    um_per_pixel : float
        Needed to recompute the anchor's own layer assignment for the left panel.

    Returns
    -------
    accepted : bool
    """
    dy, dx = shift
    Ly, Lx = fullframe_img.shape
    vmin, vmax = np.percentile(fullframe_img, 2), np.percentile(fullframe_img, 98)

    # Anchor's own layer assignment on the averaged reference (no shift needed --
    # curve_fn is native to this frame), just for the left reference panel.
    anchor_layer_cells, anchor_boundary_offsets = identify_layers_from_curve(
        med_coords_anchor, curve_fn, um_per_pixel=um_per_pixel
    )

    fig, (ax_ref, ax_layers) = plt.subplots(1, 2, figsize=(16, 8))
    try:
        fig.canvas.manager.set_window_title(f'LAYER REVIEW (propagated) -- {session_label} -- y/n')
    except Exception:
        pass

    # --- Left: averaged reference (the original shared click) ---
    avg_vmin, avg_vmax = np.percentile(averaged_image, 2), np.percentile(averaged_image, 98)
    ax_ref.imshow(averaged_image, cmap='gray', vmin=avg_vmin, vmax=avg_vmax)
    for layer_name, cell_idx in anchor_layer_cells.items():
        if len(cell_idx) == 0:
            continue
        color = _LAYER_COLORS.get(layer_name, 'white')
        ax_ref.scatter(med_coords_anchor[cell_idx, 1], med_coords_anchor[cell_idx, 0],
                       s=4, alpha=0.85, color=color, label=f'{layer_name} ({len(cell_idx)})')
    x_line_ref = np.linspace(0, averaged_image.shape[1], 200)
    ax_ref.plot(x_line_ref, curve_fn(x_line_ref), '--', color='yellow', linewidth=1.5,
                label='L4 center (shared click)')
    ax_ref.plot(x_line_ref, curve_fn(x_line_ref) + anchor_boundary_offsets['L4_upper'],
                '--', color='white', linewidth=1.0)
    ax_ref.plot(x_line_ref, curve_fn(x_line_ref) + anchor_boundary_offsets['L4_lower'],
                '--', color='white', linewidth=1.0)
    ax_ref.plot(x_line_ref, curve_fn(x_line_ref) + anchor_boundary_offsets['L5_lower'],
                '--', color='white', linewidth=1.0)
    if shared_points is not None and len(shared_points) > 0:
        ax_ref.scatter(shared_points[:, 0], shared_points[:, 1], s=60, c='red', marker='x', zorder=10)
    ax_ref.set_xlim(0, averaged_image.shape[1])
    ax_ref.set_ylim(averaged_image.shape[0], 0)
    ax_ref.set_title('AVERAGED reference (where the shared curve was clicked)')
    ax_ref.axis('off')
    ax_ref.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

    # --- Right: this session's FOV + propagated layers ---
    ax_layers.imshow(fullframe_img, cmap='gray', vmin=vmin, vmax=vmax)
    for layer_name, cell_idx in layer_cells.items():
        if len(cell_idx) == 0:
            continue
        color = _LAYER_COLORS.get(layer_name, 'white')
        ax_layers.scatter(med_coords[cell_idx, 1], med_coords[cell_idx, 0],
                           s=4, alpha=0.85, color=color, label=f'{layer_name} ({len(cell_idx)})')

    x_line = np.linspace(0, Lx, 200)
    # Boundary transformed back into this session's own raw frame:
    # depth_rel = 0  <=>  y_raw = curve_fn(x_raw + dx) - dy
    ax_layers.plot(x_line, curve_fn(x_line + dx) - dy, '--', color='yellow', linewidth=1.5,
                    label='L4 center (propagated)')
    ax_layers.plot(x_line, curve_fn(x_line + dx) - dy + boundary_offsets['L4_upper'], '--',
                    color='white', linewidth=1.2, label='L2/3-L4')
    ax_layers.plot(x_line, curve_fn(x_line + dx) - dy + boundary_offsets['L4_lower'], '--',
                    color='white', linewidth=1.2, label='L4-L5')
    ax_layers.plot(x_line, curve_fn(x_line + dx) - dy + boundary_offsets['L5_lower'], '--',
                    color='white', linewidth=1.2, label='L5-L6')

    if shared_points is not None and len(shared_points) > 0:
        points_raw = shared_points.copy()
        points_raw[:, 0] = points_raw[:, 0] - dx  # x
        points_raw[:, 1] = points_raw[:, 1] - dy  # y
        ax_layers.scatter(points_raw[:, 0], points_raw[:, 1], s=60, c='red', marker='x',
                           label='shared points (transformed)', zorder=10)

    ax_layers.set_xlim(0, Lx)
    ax_layers.set_ylim(Ly, 0)
    ax_layers.set_title(f'{session_label} -- propagated layer assignment')
    ax_layers.axis('off')
    ax_layers.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

    decision = {'accepted': None}

    def on_key(event):
        if event.key == 'y':
            decision['accepted'] = True
            fig.canvas.stop_event_loop()
        elif event.key == 'n':
            decision['accepted'] = False
            fig.canvas.stop_event_loop()

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.suptitle(f"{session_label}: does the propagated line (right) look consistent with the "
                 "shared click (left)?\n"
                 "'y' = yes, use this propagated line   |   "
                 "'n' = no, let me nudge it to the right depth for this session",
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.show(block=False)

    while decision['accepted'] is None and plt.fignum_exists(fig.number):
        fig.canvas.start_event_loop(0.1)

    if plt.fignum_exists(fig.number):
        plt.close(fig)

    if decision['accepted'] is None:
        print("Window closed without pressing y/n -- treating as reject "
              "(falling back to the adjustment window).")
        return False

    return decision['accepted']


def adjust_layer_curve_popup(fullframe_img, med_coords, curve_fn, boundary_offsets,
                                   shift, um_per_pixel, session_label,
                                   step_small=1.0, step_large=10.0):
    """
    Interactive VERTICAL-ONLY adjustment for a curve that isn't quite right
    yet -- either a rejected PROPAGATED boundary (call with this session's
    real chained shift), or a fresh independent click you're not confident
    about the exact position of (call with shift=(0.0, 0.0), since
    eyeballing precise placement while clicking is hard -- nudging
    afterward is much faster than repeatedly re-clicking from scratch).
    Keeps the curve's slope fixed and lets you nudge its vertical position
    until it sits on the real L4 band -- correcting depth positioning
    without reintroducing the slope divergence a full independent re-click
    risks (a different slope agrees near wherever you happened to click,
    but diverges more the further from that point -- see the mismatch-rate
    discussion this function exists to fix).

    Live-updates the boundary lines and per-cell layer coloring on every
    keypress so you can watch the assignment settle as you nudge.

    Controls
    --------
    Up / Down         -- nudge by step_small pixels
    Shift+Up / Down   -- nudge by step_large pixels (coarse positioning)
    'r'               -- reset the nudge back to zero (original starting position)
    'y'               -- confirm the current position
    'f'               -- give up on nudging -- fall back to a full
                         independent re-click instead (only for the rare
                         case where the slope itself, not just the
                         vertical position, genuinely needs to change)

    Parameters
    ----------
    fullframe_img : numpy.ndarray, shape (Ly, Lx)
        This session's own padded max-projection (unshifted).
    med_coords : numpy.ndarray, shape (n_cells, 2)
    curve_fn : callable
        The shared curve (anchor frame) -- slope is never modified here.
    boundary_offsets : dict
        {'L4_upper', 'L4_lower', 'L5_lower'} -- fixed anatomical offsets,
        never adjusted here, only carried along with the nudged curve.
    shift : tuple of float
        (dy, dx) for this session, from compute_chained_shifts -- the
        starting point for the nudge.
    um_per_pixel : float
    session_label : str
    step_small, step_large : float
        Nudge step sizes in pixels.

    Returns
    -------
    outcome : str
        'adjusted' or 'full_reclick'.
    final_shift : tuple of float
        (dy, dx) actually used -- dx unchanged, dy = original dy + total
        vertical nudge. Only meaningful if outcome == 'adjusted'.
    layer_cells, boundary_offsets
        Final layer assignment at the accepted position (if 'adjusted';
        otherwise the caller should discard these and do a full re-click).
    """
    dy0, dx = shift
    Ly, Lx = fullframe_img.shape
    vmin, vmax = np.percentile(fullframe_img, 2), np.percentile(fullframe_img, 98)
    x_line = np.linspace(0, Lx, 200)

    state = {'nudge_dy': 0.0, 'action': None}

    def current_shift():
        return (dy0 + state['nudge_dy'], dx)

    def current_layers():
        return identify_layers_from_shifted_curve(
            med_coords, curve_fn, current_shift(), um_per_pixel=um_per_pixel, verbose=False
        )

    fig, ax = plt.subplots(figsize=(9, 9))
    try:
        fig.canvas.manager.set_window_title(f'ADJUST -- {session_label} -- Up/Down nudge, y=confirm, f=full re-click')
    except Exception:
        pass

    def redraw():
        ax.clear()
        ax.imshow(fullframe_img, cmap='gray', vmin=vmin, vmax=vmax)

        layer_cells, _ = current_layers()
        for layer_name, cell_idx in layer_cells.items():
            if len(cell_idx) == 0:
                continue
            color = _LAYER_COLORS.get(layer_name, 'white')
            ax.scatter(med_coords[cell_idx, 1], med_coords[cell_idx, 0],
                       s=4, alpha=0.85, color=color, label=f'{layer_name} ({len(cell_idx)})')

        dy, dxc = current_shift()
        y_center = curve_fn(x_line + dxc) - dy
        ax.plot(x_line, y_center, '--', color='yellow', linewidth=1.5, label='L4 center (adjusted)')
        ax.plot(x_line, y_center + boundary_offsets['L4_upper'], '--', color='white', linewidth=1.2)
        ax.plot(x_line, y_center + boundary_offsets['L4_lower'], '--', color='white', linewidth=1.2)
        ax.plot(x_line, y_center + boundary_offsets['L5_lower'], '--', color='white', linewidth=1.2)

        ax.set_xlim(0, Lx)
        ax.set_ylim(Ly, 0)
        ax.set_title(f"{session_label} -- vertical nudge = {state['nudge_dy']:+.1f}px "
                     f"({state['nudge_dy'] * um_per_pixel:+.1f}um)", fontsize=11)
        ax.axis('off')
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'up':
            state['nudge_dy'] += step_small
            redraw()
        elif event.key == 'down':
            state['nudge_dy'] -= step_small
            redraw()
        elif event.key == 'shift+up':
            state['nudge_dy'] += step_large
            redraw()
        elif event.key == 'shift+down':
            state['nudge_dy'] -= step_large
            redraw()
        elif event.key == 'r':
            state['nudge_dy'] = 0.0
            redraw()
        elif event.key == 'y':
            state['action'] = 'adjusted'
            fig.canvas.stop_event_loop()
        elif event.key == 'f':
            state['action'] = 'full_reclick'
            fig.canvas.stop_event_loop()

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.suptitle("Up/Down = nudge   Shift+Up/Down = big nudge   'r' = reset\n"
                 "'y' = confirm this position   'f' = give up, let me click a fresh boundary instead",
                 fontsize=12, fontweight='bold')
    redraw()
    plt.tight_layout()
    plt.show(block=False)

    while state['action'] is None and plt.fignum_exists(fig.number):
        fig.canvas.start_event_loop(0.1)

    if plt.fignum_exists(fig.number):
        plt.close(fig)

    if state['action'] is None:
        print("Window closed without a decision -- treating as full re-click fallback.")
        state['action'] = 'full_reclick'

    final_shift = current_shift()
    layer_cells, boundary_offsets_final = current_layers()

    if state['action'] == 'adjusted':
        print(f"  {session_label}: accepted with vertical nudge = {state['nudge_dy']:+.1f}px "
              f"({state['nudge_dy'] * um_per_pixel:+.1f}um)")

    return state['action'], final_shift, layer_cells, boundary_offsets_final


# ============================================================
# NEW: per-session driver (propagate, confirm, override, save)
# ============================================================
def process_session_layer_curve_propagated(session_dir, shift, curve_fn, shared_points,
                                            um_per_pixel, averaged_image, med_coords_anchor,
                                            session_label=None, degree=1):
    """
    One session's Phase 1 output via the propagate-then-confirm workflow:
    evaluate the shared curve at this session's own cells (via its chained
    shift), show the propagated result for confirmation. If rejected, opens
    a vertical-only adjustment window (adjust_layer_curve_popup) that
    keeps the shared slope and lets you nudge for real per-session depth
    drift; only falls back to a full independent re-click
    (pick_and_confirm_layers) if you explicitly give up on nudging ('f').

    Parameters
    ----------
    session_dir : str
        Path to the TSeries folder (contains suite2p/plane0/).
    shift : tuple of float
        (dy, dx) for this session, from compute_chained_shifts.
    curve_fn : callable
        The one shared curve, fit on the averaged reference.
    shared_points : numpy.ndarray
        The shared clicked points (anchor frame) -- shown, transformed,
        in the confirm popup for context. Saved as-is if propagation is
        accepted (this session's own points if overridden).
    um_per_pixel : float
    averaged_image : numpy.ndarray
        The averaged reference image, passed through to the confirm popup
        so it always has the original shared click to refer back to.
    med_coords_anchor : numpy.ndarray
        Anchor session's own cells, for the confirm popup's reference panel.
    session_label : str, optional
    degree : int

    Returns
    -------
    dict with keys: session_label, save_path, fullframe_img, med_coords,
    cell_idx, points, coeffs, curve_fn, layer_cells, boundary_offsets, source.
    """
    if session_label is None:
        session_label = os.path.basename(session_dir)

    plane0_path = os.path.join(session_dir, 'suite2p', 'plane0')
    ops = np.load(os.path.join(plane0_path, 'ops.npy'), allow_pickle=True).item()
    stat = np.load(os.path.join(plane0_path, 'stat.npy'), allow_pickle=True)
    iscell = np.load(os.path.join(plane0_path, 'iscell.npy'), allow_pickle=True)

    cell_idx = np.where(iscell[:, 0] == 1)[0]
    med_coords = np.array([stat[i]['med'] for i in cell_idx])
    fullframe_img = get_fullframe_max_proj(ops)

    print(f"\n{session_label}: {len(cell_idx)} cells  (shift=({shift[0]:+.2f}, {shift[1]:+.2f}))")

    layer_cells, boundary_offsets = identify_layers_from_shifted_curve(
        med_coords, curve_fn, shift, um_per_pixel=um_per_pixel
    )

    accepted = review_propagated_layers_popup(
        fullframe_img, med_coords, layer_cells, curve_fn, boundary_offsets,
        shift, shared_points, session_label, averaged_image, med_coords_anchor, um_per_pixel
    )

    extra_attrs = None

    if accepted:
        source = 'propagated'
        points_used, coeffs_used, curve_fn_used = shared_points, curve_fn.coefficients, curve_fn
    else:
        print(f"  Propagated boundary rejected for {session_label} -- opening vertical-adjustment window.")
        outcome, adjusted_shift, adjusted_layer_cells, adjusted_boundary_offsets = adjust_layer_curve_popup(
            fullframe_img, med_coords, curve_fn, boundary_offsets, shift, um_per_pixel, session_label
        )

        if outcome == 'adjusted':
            source = 'propagated_adjusted'
            points_used, coeffs_used, curve_fn_used = shared_points, curve_fn.coefficients, curve_fn
            layer_cells, boundary_offsets = adjusted_layer_cells, adjusted_boundary_offsets
            extra_attrs = {'vertical_nudge_px': adjusted_shift[0] - shift[0]}
        else:
            print(f"  Falling back to a full independent re-click for {session_label}.")
            points_used, coeffs_used, curve_fn_used, layer_cells, boundary_offsets = pick_and_confirm_layers(
                fullframe_img, med_coords, um_per_pixel=um_per_pixel, degree=degree,
                reference_curve_fn=curve_fn, reference_shift=shift
            )
            source = 'overridden'

    layer_codes, layer_names = layer_cells_to_labels(layer_cells, len(med_coords))

    # Distinct filename from the earlier independent-click version's output --
    # never overwrites the independent-click results already computed for real.
    save_path = os.path.join(session_dir, f'{session_label}_layer_curve_results_averaged.h5')
    save_layer_curve_results(
        save_path, session_label, cell_idx, layer_codes, layer_names,
        points_used, coeffs_used, degree, um_per_pixel, boundary_offsets, source=source,
        extra_attrs=extra_attrs
    )

    return {
        'session_label': session_label,
        'save_path': save_path,
        'fullframe_img': fullframe_img,
        'med_coords': med_coords,
        'cell_idx': cell_idx,
        'points': points_used,
        'coeffs': coeffs_used,
        'curve_fn': curve_fn_used,
        'layer_cells': layer_cells,
        'boundary_offsets': boundary_offsets,
        'source': source,
    }


# ============================================================
# NEW: per-animal batch driver (the new default entry point)
# ============================================================
def process_animal_layer_curve_averaged(animal_cfg):
    """
    New default per-animal driver: chained registration across all of the
    animal's sessions -> one denoised averaged reference -> ONE shared
    click/fit -> per-session propagate + confirm (+ manual override where
    rejected) -> save.

    Parameters
    ----------
    animal_cfg : dict
        Keys: 'animal_id', 'store_dir', 'sessions' (list of session_dir
        paths, ANY order -- sorted chronologically internally),
        'um_per_pixel', 'degree' (optional, default 1).

    Returns
    -------
    session_labels, fullframe_imgs, curve_results, shifts
    """
    animal_id = animal_cfg['animal_id']
    store_dir = animal_cfg['store_dir']
    um_per_pixel = animal_cfg['um_per_pixel']
    degree = animal_cfg.get('degree', 1)
    sessions = [s for s in animal_cfg['sessions'] if os.path.isdir(s)]

    print("\n" + "=" * 90)
    print(f" PHASE 1 LAYER ASSIGNMENT (averaged/chained): {animal_id}")
    print("=" * 90)

    sessions_sorted = sort_sessions_chronologically(sessions)
    print("\nChronological order:")
    for s in sessions_sorted:
        print(f"  {_extract_session_datetime(s)}  {os.path.basename(s)}")

    print("\nLoading full-frame images for chained registration...")
    fullframe_imgs_ordered = {}
    for session_dir in sessions_sorted:
        label = os.path.basename(session_dir)
        plane0_path = os.path.join(session_dir, 'suite2p', 'plane0')
        ops = np.load(os.path.join(plane0_path, 'ops.npy'), allow_pickle=True).item()
        fullframe_imgs_ordered[label] = get_fullframe_max_proj(ops)

    print("\nComputing chained shifts (each session registered to its temporal neighbor)...")
    shifts = compute_chained_shifts(fullframe_imgs_ordered)

    print("\nBuilding averaged reference image...")
    averaged_image = build_averaged_reference(fullframe_imgs_ordered, shifts)

    anchor_dir = sessions_sorted[0]
    anchor_label = os.path.basename(anchor_dir)
    plane0_anchor = os.path.join(anchor_dir, 'suite2p', 'plane0')
    stat_anchor = np.load(os.path.join(plane0_anchor, 'stat.npy'), allow_pickle=True)
    iscell_anchor = np.load(os.path.join(plane0_anchor, 'iscell.npy'), allow_pickle=True)
    cell_idx_anchor = np.where(iscell_anchor[:, 0] == 1)[0]
    med_coords_anchor = np.array([stat_anchor[i]['med'] for i in cell_idx_anchor])

    print(f"\nClick the L4 band ONCE on the averaged reference (anchor: {anchor_label})...")
    points, coeffs, curve_fn, _, _ = pick_and_confirm_layers(
        averaged_image, med_coords_anchor, um_per_pixel=um_per_pixel, degree=degree
    )

    print(f"\nPropagating shared curve to all {len(sessions_sorted)} sessions...")
    session_labels, fullframe_imgs, curve_results = [], [], {}
    for session_dir in sessions_sorted:
        label = os.path.basename(session_dir)
        result = process_session_layer_curve_propagated(
            session_dir, shift=shifts[label], curve_fn=curve_fn, shared_points=points,
            um_per_pixel=um_per_pixel, averaged_image=averaged_image,
            med_coords_anchor=med_coords_anchor, session_label=label, degree=degree
        )
        session_labels.append(result['session_label'])
        fullframe_imgs.append(result['fullframe_img'])
        curve_results[result['session_label']] = load_layer_curve_results(result['save_path'])

    if len(session_labels) >= 2:
        fig_path = os.path.join(store_dir, f'{animal_id}_curve_boundary_consistency_averaged.png')
        plot_curve_boundary_consistency(session_labels, fullframe_imgs, curve_results,
                                         save_path=fig_path)

    n_propagated = sum(1 for r in curve_results.values() if r.get('source') == 'propagated')
    n_overridden = sum(1 for r in curve_results.values() if r.get('source') == 'overridden')
    print(f"\nDone. {n_propagated}/{len(session_labels)} sessions used the propagated curve, "
          f"{n_overridden} were manually overridden.")

    return session_labels, fullframe_imgs, curve_results, shifts


# ============================================================
# Entry point
# ============================================================
if __name__ == '__main__':

    # Same session lists as 1.LayerAssignment_Curve.py's ANIMALS_LAYER_CURVE --
    # order doesn't matter here, sort_sessions_chronologically handles it.
    ANIMALS_LAYER_CURVE = [
        # {
        #     'animal_id': 'JSY090',
        #     'store_dir': r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD',
        #     'um_per_pixel': 1.08952017715202,
        #     'degree': 1,
        #     'sessions': [
        #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260719_JSY_JSY090_LongitudinalImaging_DREADD_Day1\TSeries-07192026-0941-001',
        #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260720_JSY_JSY090_LongitudinalImaging_DREADD_Day2\TSeries-07202026-1009-001',
        #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260721_JSY_JSY090_LongitudinalImaging_DREADD_Day3\TSeries-07212026-0907-001',
        #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260722_JSY_JSY090_LongitudinalImaging_DREADD_Day4\TSeries-07222026-1831-002',
        #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260723_JSY_JSY090_LongitudinalImaging_DREADD_Day5\TSeries-07232026-0843-001',

        #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260724_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_1\TSeries-07242026-0809_DCZ-001',
        #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260724_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_1\TSeries-07242026-0809_SALINE-001',
        #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260726_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_2\TSeries-07262026-0755_SAL-001',
        #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260726_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_2\TSeries-07262026-0755_DCZ-001',
        #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260728_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_3\TSeries-07282026-0811_SAL-001',
        #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260728_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_3\TSeries-07282026-0811_DCZ-001',

        #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260730_JSY_JSY090_LongitudinalImaging_DREADD_ActiveOpenLoop_Saline_DCZ\TSeries-07302026-0844_SAL-001',
        #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260730_JSY_JSY090_LongitudinalImaging_DREADD_ActiveOpenLoop_Saline_DCZ\TSeries-07302026-0844_DCZ-001',
        #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260801_JSY_JSY090_LongitudinalImaging_DREADD_StationaryOpenLoop_Saline_DCZ\TSeries-08012026-0815_sal-002',
        #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260801_JSY_JSY090_LongitudinalImaging_DREADD_StationaryOpenLoop_Saline_DCZ\TSeries-08012026-0815_DCZ-001',
        #     ],
        # },
        {
            'animal_id': 'JSY093',
            'store_dir': r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD',
            'um_per_pixel': 1.08952017715202,
            'degree': 1,
            'sessions': [
                r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260719_JSY_JSY093_LongitudinalImaging_DREADD_Day1\TSeries-07192026-0941-001',
                r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260720_JSY_JSY093_LongitudinalImaging_DREADD_Day2\TSeries-07202026-1009-001',
                r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260721_JSY_JSY093_LongitudinalImaging_DREADD_Day3\TSeries-07212026-0907-001',
                r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260722_JSY_JSY093_LongitudinalImaging_DREADD_Day4\TSeries-07222026-0959-001',
                r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260723_JSY_JSY093_LongitudinalImaging_DREADD_Day5\TSeries-07232026-0843-001',

                r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260724_JSY_JSY093_LongitudinalImaging_DREADD_Saline_DCZ_1\TSeries-07242026-0809_DCZ-001',
                r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260724_JSY_JSY093_LongitudinalImaging_DREADD_Saline_DCZ_1\TSeries-07242026-0809_SALINE-001',
                r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260726_JSY_JSY093_LongitudinalImaging_DREADD_Saline_DCZ_2\TSeries-07262026-0755_DCZ-001',
                r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260726_JSY_JSY093_LongitudinalImaging_DREADD_Saline_DCZ_2\TSeries-07262026-0755-001',
                r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260728_JSY_JSY093_LongitudinalImaging_DREADD_Saline_DCZ_3\TSeries-07282026-0811_DCZ-001',
                r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260728_JSY_JSY093_LongitudinalImaging_DREADD_Saline_DCZ_3\TSeries-07282026-0811_SAL-001',

                r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260730_JSY_JSY093_LongitudinalImaging_DREADD_ActiveOpenLoop_Saline_DCZ\TSeries-07302026-0844_SAL-001',
                r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260730_JSY_JSY093_LongitudinalImaging_DREADD_ActiveOpenLoop_Saline_DCZ\TSeries-07302026-0844_DCZ-001',
                r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260801_JSY_JSY093_LongitudinalImaging_DREADD_StationaryOpenLoop_Saline_DCZ\TSeries-08012026-0815_sal-001',
                r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260801_JSY_JSY093_LongitudinalImaging_DREADD_StationaryOpenLoop_Saline_DCZ\TSeries-08012026-0815_dcz-001',
            ],
        },
    ]

    animal_results = {}
    for animal_cfg in ANIMALS_LAYER_CURVE:
        animal_results[animal_cfg['animal_id']] = process_animal_layer_curve_averaged(animal_cfg)

        session_labels, fullframe_imgs, curve_results, shifts = animal_results[animal_cfg['animal_id']]
        if len(session_labels) >= 2:
            summarize_layer_curve_consistency(session_labels, curve_results)
