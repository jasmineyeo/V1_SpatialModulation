"""
helper/LayerBoundaryParams.py
Per-animal persistence and cross-session QC for manually-picked layer
boundaries (rotation_deg + L4 peak_y), used for FOVs where the automatic
identify_layers() pipeline is unreliable (tilted FOV, patchy density).

Each session's picks are stored independently (no assumption that tilt is
stable across a longitudinal series) and can be visually cross-checked with
plot_boundary_consistency before trusting them for layer-resolved analysis.

JSY, 2026
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation


# ============================================================
# Load / save per-animal parameter store
# ============================================================
def load_params(store_path):
    """Load the per-animal {session_label: {rotation_deg, peak_y}} store."""
    if not os.path.exists(store_path):
        return {}
    with open(store_path, 'r') as f:
        return json.load(f)


def save_param(store_path, session_label, rotation_deg, peak_y):
    """
    Persist one session's picked rotation_deg/peak_y into the per-animal
    JSON store (creating it if needed). Returns the full updated dict.
    """
    params = load_params(store_path)
    params[session_label] = {
        'rotation_deg': float(rotation_deg),
        'peak_y': float(peak_y),
    }
    store_dir = os.path.dirname(store_path)
    if store_dir:
        os.makedirs(store_dir, exist_ok=True)
    with open(store_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Saved layer-boundary params for '{session_label}' -> {store_path}")
    return params


# ============================================================
# Cross-session consistency QC
# ============================================================
def _boundary_line_xy(mean_img, rotation_deg, peak_y):
    """
    Compute the (x, y) endpoints of the picked L4 centerline in raw image
    space, for a line rotated by rotation_deg about the image center and
    passing through depth_y == peak_y (same convention as
    SpatialModulationIndexLayerSpecific.identify_layers, but using the image
    center rather than the cell centroid — an approximation that's fine for
    visual QC).
    """
    Ly, Lx = mean_img.shape
    cy, cx = Ly / 2.0, Lx / 2.0
    theta = np.radians(rotation_deg)

    x_vals = np.array([0, Lx])
    if abs(np.cos(theta)) < 1e-8:
        # Vertical line (90 degree rotation) — degenerate for this FOV geometry
        y_vals = np.array([0, Ly])
        x_vals = np.array([peak_y, peak_y])
    else:
        y_vals = cy + ((peak_y - cy) + (x_vals - cx) * np.sin(theta)) / np.cos(theta)

    return x_vals, y_vals


def plot_boundary_consistency(session_labels, mean_imgs, params, reference_label=None,
                               save_path=None):
    """
    Overlay each session's picked L4 boundary line on a common reference
    frame, so drift or outlier picks are visible before trusting them.

    Parameters:
    -----------
    session_labels : list of str
        Session labels, matching keys in `params`.
    mean_imgs : list of numpy.ndarray
        Mean FOV image per session, same order as session_labels.
    params : dict
        {session_label: {'rotation_deg': ..., 'peak_y': ...}}, e.g. from
        load_params().
    reference_label : str, optional
        Session to use as the alignment reference. Defaults to the first
        session with an entry in `params`.
    save_path : str, optional
        If given, saves the figure there.

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    available = [s for s in session_labels if s in params]
    if len(available) == 0:
        raise ValueError("None of the given session_labels have entries in params")

    if reference_label is None:
        reference_label = available[0]
    if reference_label not in available:
        raise ValueError(f"reference_label '{reference_label}' has no entry in params")

    label_to_img = dict(zip(session_labels, mean_imgs))
    ref_img = label_to_img[reference_label]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(ref_img, cmap='gray',
              vmin=np.percentile(ref_img, 2), vmax=np.percentile(ref_img, 98))

    colors = plt.cm.hsv(np.linspace(0, 0.85, len(available)))

    for color, label in zip(colors, available):
        img = label_to_img[label]
        p = params[label]

        if label == reference_label:
            dy, dx = 0.0, 0.0
        else:
            shift_yx, _, _ = phase_cross_correlation(ref_img, img, upsample_factor=10)
            dy, dx = shift_yx[0], shift_yx[1]

        x_vals, y_vals = _boundary_line_xy(img, p['rotation_deg'], p['peak_y'])
        ax.plot(x_vals + dx, y_vals + dy, '-', color=color, linewidth=2,
                 label=f"{label} (rot={p['rotation_deg']:.1f}\N{DEGREE SIGN}, "
                       f"peak_y={p['peak_y']:.0f})")

    ax.set_title(f"L4 boundary consistency across sessions\n(reference: {reference_label})")
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Boundary consistency figure saved to {save_path}")

    return fig
