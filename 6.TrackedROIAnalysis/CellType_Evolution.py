"""
CellType_Evolution.py
=====================
Tracked-cell functional cell type evolution analysis.

Follows the same neurons across all sessions to ask:
  1. Does each cell's functional type change across experience?
  2. Do type proportions differ by layer?
  3. Do experience-dependent changes in proportions differ by layer?

Approach:
  - Re-fits the PCA + k-means model on the population data from *_pca_data.h5
    (same procedure as 4.PCA/PCA_ComprehensiveAnalysis.py, ensuring consistency)
  - For each tracked cell × session, computes its spatial profile,
    applies the same preprocessing (trim → interpolate → z-score),
    then projects through PCA and assigns to the nearest cluster
  - Builds label_tensor (n_tracked, n_sessions) of per-cell type labels
  - Layer assignments from the reference session via assign_layers_from_smi

Prerequisite: run 4.PCA/PCA_DataAggregation.py first to produce *_pca_data.h5.

JSY, 2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d as scipy_interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import chi2_contingency, friedmanchisquare, kruskal, linregress
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from load_tracked import (
    load_tracking, find_files_from_tracking, find_smi_files, find_preproc_files,
    assign_layers_from_smi, load_preproc_session,
    parse_day_numbers, animal_id_from_path,
    LAYER_ORDER, LAYER_COLORS, report_found_files,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Population PCA data file — produced by 4.PCA/PCA_DataAggregation.py
# The PCA + k-means model is re-derived from this file for consistency.
PCA_DATA_FILE     = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\PCA\JSY054_pca_data.h5"

# Tracked ROI file — produced by 6.TrackedROIAnalysis/TrackROIs.ipynb
ROI_TRACKING_FILE = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\roi_tracking_results.h5"
ANIMAL_DIR        = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging"
REFERENCE_DAY     = "Day2"   # used for layer assignment

# Profile preprocessing — must match PCA_DataAggregation.py settings
TRIM_START_CM  = 10.0    # cm — start of analysis window
TRIM_END_CM    = 125.0   # cm — end of analysis window
TARGET_N_BINS  = 115     # bins after interpolation
SMOOTH_SIGMA   = 1.0     # Gaussian smoothing before trimming

# PCA / clustering — must match 4.PCA/PCA_ComprehensiveAnalysis.py
N_PCA_COMPONENTS = 10
N_CLUSTER_PCS    = 5
K_RANGE          = range(2, 8)
OVERRIDE_K       = None   # set to int to force a specific k

OUTPUT_DIR = None   # None → same folder as this script

# ============================================================


# ── Profile preprocessing ─────────────────────────────────────

def preprocess_profiles(nsa, bc, reliable_cells,
                         trim_start, trim_end, target_n_bins,
                         smooth_sigma=1.0):
    """
    Convert raw norm_spatial_activity to the same feature space used by
    PCA_DataAggregation.py:
      1. Mean across trials
      2. Gaussian smooth
      3. Trim to [trim_start, trim_end] cm  (bc is in actual cm)
      4. Interpolate to target_n_bins on linspace(trim_start, trim_end)
      5. Z-score per cell (shape normalisation)

    Parameters
    ----------
    nsa            : (n_cells, n_trials, n_bins)
    bc             : (n_bins,) — bin centers in cm
    reliable_cells : (n_cells,) bool
    trim_start/end : float — cm boundaries
    target_n_bins  : int
    smooth_sigma   : float

    Returns
    -------
    profiles_out : (n_cells, target_n_bins) — NaN for unreliable / bad cells
    target_bc    : (target_n_bins,) — common bin centers after interpolation
    """
    n_cells  = nsa.shape[0]
    target_bc = np.linspace(trim_start, trim_end, target_n_bins)
    profiles_out = np.full((n_cells, target_n_bins), np.nan)

    # Mean across trials
    mean_profiles = np.nanmean(nsa, axis=1).astype(float)   # (n_cells, n_bins)

    for c in range(n_cells):
        if not reliable_cells[c]:
            continue

        profile = gaussian_filter1d(mean_profiles[c], sigma=smooth_sigma)

        # Trim to [trim_start, trim_end] using actual cm bin_centers
        trim_mask = (bc >= trim_start) & (bc <= trim_end)
        if trim_mask.sum() < 3:
            continue   # not enough bins in window

        bc_trim  = bc[trim_mask]
        pr_trim  = profile[trim_mask]

        # Interpolate to common grid
        # Use edge extrapolation (not NaN) to match PCA_DataAggregation.py —
        # target_bc endpoints (e.g. 10.0, 125.0) can lie just outside the
        # session bc_trim range (e.g. 10.5–124.5 for 1 cm bins), which would
        # produce NaN fill values and incorrectly invalidate every cell.
        f = scipy_interp1d(bc_trim, pr_trim, kind='linear',
                           bounds_error=False,
                           fill_value=(pr_trim[0], pr_trim[-1]))
        interp_profile = f(target_bc)

        # Z-score per cell
        std = np.nanstd(interp_profile)
        if std > 0:
            profiles_out[c] = (interp_profile - np.nanmean(interp_profile)) / std
        # else: leave as NaN (flat profile)

    return profiles_out, target_bc


# ── Model fitting (mirrors population script) ─────────────────

def load_pca_data_for_model(filepath):
    """
    Load z-scored profiles from *_pca_data.h5 to refit the PCA + KMeans model.
    Also returns bin_centers, landmark_positions, and cell labels.
    """
    with h5py.File(filepath, 'r') as f:
        if 'features/spatial_profiles_session_corrected' in f:
            profiles = f['features/spatial_profiles_session_corrected'][:]
        else:
            profiles = f['features/spatial_profiles_zscore'][:]

        bin_centers        = f['metadata/bin_centers_trimmed'][:]
        landmark_positions = f['metadata/landmark_positions'][:]
        animal_id          = f['metadata'].attrs['animal_id']

    animal_id = animal_id if isinstance(animal_id, str) else animal_id.decode()
    print(f"  Population model source: {animal_id}  |  {len(profiles)} cells")
    return profiles, bin_centers, landmark_positions


def fit_model(profiles, n_pca, n_cluster_pcs, k_range, override_k=None,
              random_state=42):
    """
    Fit PCA + KMeans on population profiles (same procedure as population script).
    Returns (pca, kmeans, optimal_k, raw_labels, centroids).
    """
    pca    = PCA(n_components=n_pca)
    scores = pca.fit_transform(profiles)
    X      = scores[:, :n_cluster_pcs]

    # K selection
    sil_scores, inertias = [], []
    for k in k_range:
        km   = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labs = km.fit_predict(X)
        sil_scores.append(silhouette_score(X, labs))
        inertias.append(km.inertia_)

    auto_k    = list(k_range)[int(np.argmax(sil_scores))]
    optimal_k = override_k if override_k is not None else auto_k
    print(f"  K selection: auto={auto_k}  used={optimal_k}")

    kmeans     = KMeans(n_clusters=optimal_k, n_init=20, random_state=random_state)
    raw_labels = kmeans.fit_predict(X)
    centroids  = kmeans.cluster_centers_

    return pca, kmeans, optimal_k, raw_labels, centroids


def assign_semantic_labels(raw_labels, profiles_pop, n_types,
                            bin_centers, landmark_positions):
    """
    Map cluster IDs to semantic names based on mean profile peak position.
    Earliest peak → Adaptation-like, latest peak → L4-preferring,
    most uniform → Visually responsive.
    """
    mean_profiles = np.array([
        np.mean(profiles_pop[raw_labels == t], axis=0)
        if np.sum(raw_labels == t) > 0 else np.zeros(profiles_pop.shape[1])
        for t in range(n_types)
    ])
    peak_bins = np.argmax(mean_profiles, axis=1)

    names = [None] * n_types
    used  = set()

    def _pick(order):
        for k in order:
            if k not in used:
                used.add(k)
                return k

    names[_pick(np.argsort(peak_bins))]         = 'Adaptation-like'
    names[_pick(np.argsort(peak_bins)[::-1])]   = 'L4-preferring'
    ranges = mean_profiles.max(axis=1) - mean_profiles.min(axis=1)
    names[_pick(np.argsort(ranges))]             = 'Visually responsive'

    for k in range(n_types):
        if names[k] is None:
            peak_cm  = bin_centers[peak_bins[k]] if peak_bins[k] < len(bin_centers) else -1
            names[k] = f'Peak~{peak_cm:.0f}cm'

    base_colors = {
        'Adaptation-like':    '#E53935',
        'L4-preferring':      '#4CAF50',
        'Visually responsive':'#1E88E5',
    }
    extra = ['#FF9800', '#9C27B0', '#00BCD4', '#795548']
    ec, colors = 0, []
    for name in names:
        colors.append(base_colors.get(name, extra[ec % len(extra)]))
        if name not in base_colors:
            ec += 1

    return names, colors, mean_profiles


# ── Tracked-cell feature extraction ──────────────────────────

def build_label_tensor(tracked_matrix, day_labels, preproc_files,
                        pca, kmeans, n_types,
                        trim_start, trim_end, target_n_bins, smooth_sigma):
    """
    For each tracked cell × session:
      1. Extract mean spatial profile from preproc HDF5
      2. Apply same preprocessing as population PCA
      3. Project through PCA.transform(), assign cluster via KMeans.predict()
      4. Store in label_tensor[row, col]

    Returns
    -------
    label_tensor : (n_tracked, n_sessions) int — cluster label or -1 if absent/bad
    """
    n_tracked, n_sessions = tracked_matrix.shape
    label_tensor = np.full((n_tracked, n_sessions), -1, dtype=int)

    for col, day in enumerate(day_labels):
        if day not in preproc_files:
            print(f"  {day}: no preproc file — skipping")
            continue

        print(f"  {day}...", end=' ', flush=True)
        _, nsa, bc, rel, _, _, _ = load_preproc_session(preproc_files[day])

        # Preprocess all cells in this session to PCA feature space
        profiles_sess, _ = preprocess_profiles(
            nsa, bc, rel, trim_start, trim_end, target_n_bins, smooth_sigma
        )
        # profiles_sess : (n_cells_session, TARGET_N_BINS) — NaN for bad cells

        # Find which tracked cells are present in this session
        roi_indices = tracked_matrix[:, col]
        valid_rows  = np.where(roi_indices >= 0)[0]

        if len(valid_rows) == 0:
            print("0 valid")
            continue

        # Collect valid profiles for batch PCA transform
        batch_profiles = []
        batch_rows     = []
        for row in valid_rows:
            roi = roi_indices[row]
            if roi >= len(profiles_sess):
                continue
            profile = profiles_sess[roi]
            if np.any(np.isnan(profile)):
                continue
            batch_profiles.append(profile)
            batch_rows.append(row)

        if len(batch_profiles) == 0:
            print("0 valid (all NaN profiles)")
            continue

        batch_profiles = np.array(batch_profiles)   # (n_valid, TARGET_N_BINS)

        # Project through population PCA, assign cluster
        n_pc       = kmeans.cluster_centers_.shape[1]
        pc_scores  = pca.transform(batch_profiles)[:, :n_pc]
        cluster_ids = kmeans.predict(pc_scores)

        for i, row in enumerate(batch_rows):
            label_tensor[row, col] = int(cluster_ids[i])

        n_labelled = len(batch_rows)
        print(f"{n_labelled} labelled")

    return label_tensor


# ── Proportion matrix ─────────────────────────────────────────

def build_prop_matrix(label_tensor, cell_layers, layer_list, n_types):
    """
    Build (n_types, n_layers, n_sessions) proportion matrix from label_tensor.
    Also returns (n_layers, n_sessions) count matrix.
    """
    n_tracked, n_sessions = label_tensor.shape
    n_layers = len(layer_list)
    prop   = np.full((n_types, n_layers, n_sessions), np.nan)
    counts = np.zeros((n_layers, n_sessions), dtype=int)

    for li, layer in enumerate(layer_list):
        if layer not in cell_layers or len(cell_layers[layer]) == 0:
            continue
        rows = cell_layers[layer]
        for col in range(n_sessions):
            col_labels = label_tensor[rows, col]
            valid      = col_labels >= 0
            n_valid    = int(np.sum(valid))
            if n_valid == 0:
                continue
            counts[li, col] = n_valid
            for t in range(n_types):
                prop[t, li, col] = np.sum(col_labels[valid] == t) / n_valid

    return prop, counts


# ── Plotting ─────────────────────────────────────────────────

def plot_label_tensor_heatmap(label_tensor, cell_layers, layer_list,
                               day_labels, type_names, type_colors,
                               animal_id, output_path=None):
    """
    Heatmap of label_tensor: rows = tracked cells (sorted by layer),
    columns = sessions, color = cell type.
    """
    n_tracked, n_sessions = label_tensor.shape
    n_types = len(type_names)

    # Sort rows by layer
    row_order  = []
    layer_boundaries = []
    for layer in layer_list:
        if layer not in cell_layers:
            continue
        rows = sorted(cell_layers[layer].tolist())
        layer_boundaries.append((len(row_order), len(row_order) + len(rows), layer))
        row_order.extend(rows)

    if not row_order:
        print("  No layer-assigned cells for heatmap")
        return None

    sorted_tensor = label_tensor[row_order, :]   # reorder rows

    # Build color image: (n_rows, n_sessions, 3) RGB
    # -1 = absent/bad → white; 0..n_types-1 → type color
    import matplotlib.colors as mcolors
    rgb = np.ones((len(row_order), n_sessions, 3))
    for t in range(n_types):
        c_rgb = mcolors.to_rgb(type_colors[t])
        mask  = sorted_tensor == t
        rgb[mask] = c_rgb
    # -1 stays white

    fig, ax = plt.subplots(figsize=(max(6, n_sessions * 1.2), max(5, len(row_order) / 30)))
    fig.suptitle(f'{animal_id} — Tracked Cell Type per Session', fontweight='bold')

    ax.imshow(rgb, aspect='auto', interpolation='none')

    # Layer boundary lines and labels
    for start, end, layer in layer_boundaries:
        ax.axhline(start - 0.5, color='black', linewidth=1.0)
        ax.text(-0.7, (start + end) / 2, layer,
                ha='right', va='center', fontsize=8,
                color=LAYER_COLORS.get(layer, 'black'), fontweight='bold',
                transform=ax.get_yaxis_transform())

    ax.set_xticks(range(n_sessions))
    ax.set_xticklabels(day_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks([])
    ax.set_ylabel('Tracked cells (sorted by layer)')

    # Legend
    handles = [mpatches.Patch(color=type_colors[t], label=type_names[t])
               for t in range(n_types)]
    handles.append(mpatches.Patch(color='white', label='Absent/unlabelled',
                                   edgecolor='gray'))
    ax.legend(handles=handles, fontsize=7, loc='upper right',
              bbox_to_anchor=(1.25, 1.0))

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_stacked_bars_by_layer(prop, counts, layer_list, day_labels,
                                type_names, type_colors, animal_id, output_path=None):
    n_layers = len(layer_list)
    n_types  = len(type_names)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4), sharey=True)
    fig.suptitle(f'{animal_id} — Cell Type Proportions per Layer (tracked)',
                 fontweight='bold')
    if n_layers == 1:
        axes = [axes]

    xpos = np.arange(len(day_labels))
    for li, (layer, ax) in enumerate(zip(layer_list, axes)):
        bottoms = np.zeros(len(day_labels))
        for t in range(n_types):
            vals    = prop[t, li, :]
            heights = np.where(~np.isnan(vals), vals * 100, 0)
            ax.bar(xpos, heights, bottom=bottoms,
                   color=type_colors[t],
                   label=type_names[t] if li == 0 else '',
                   width=0.7, alpha=0.85)
            bottoms += heights
        for xi in range(len(day_labels)):
            n = counts[li, xi]
            if n > 0:
                ax.text(xi, 103, f'n={n}', ha='center', va='bottom',
                        fontsize=6, rotation=45)
        ax.set_title(layer, color=LAYER_COLORS.get(layer, 'black'), fontweight='bold')
        ax.set_xticks(xpos)
        ax.set_xticklabels(day_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 120)
        ax.set_ylabel('% cells' if li == 0 else '')
        ax.grid(True, alpha=0.2, axis='y')

    handles = [mpatches.Patch(color=type_colors[t], label=type_names[t])
               for t in range(n_types)]
    axes[-1].legend(handles=handles, fontsize=7, loc='upper right')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_trajectories_by_layer(prop, layer_list, session_days, day_labels,
                                type_names, type_colors, animal_id, output_path=None):
    n_layers = len(layer_list)
    n_types  = len(type_names)
    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4), sharey=True)
    fig.suptitle(f'{animal_id} — Cell Type Trajectory per Layer (tracked)',
                 fontweight='bold')
    if n_layers == 1:
        axes = [axes]

    x = np.array(session_days)
    for li, (layer, ax) in enumerate(zip(layer_list, axes)):
        for t in range(n_types):
            y     = prop[t, li, :] * 100
            valid = ~np.isnan(y)
            if valid.sum() < 2:
                continue
            ax.plot(x[valid], y[valid], 'o-', color=type_colors[t],
                    label=type_names[t], linewidth=2, markersize=5)
        ax.set_title(layer, color=LAYER_COLORS.get(layer, 'black'), fontweight='bold')
        ax.set_xlabel('Recording Day')
        ax.set_ylabel('% cells' if li == 0 else '')
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(day_labels, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_trajectories_by_type(prop, layer_list, session_days, day_labels,
                               type_names, type_colors, animal_id, output_path=None):
    n_types = len(type_names)
    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4), sharey=True)
    fig.suptitle(f'{animal_id} — Layer Trajectories per Cell Type (tracked)',
                 fontweight='bold')
    if n_types == 1:
        axes = [axes]

    x = np.array(session_days)
    for t, ax in enumerate(axes):
        for li, layer in enumerate(layer_list):
            y     = prop[t, li, :] * 100
            valid = ~np.isnan(y)
            if valid.sum() < 2:
                continue
            ax.plot(x[valid], y[valid], 'o-',
                    color=LAYER_COLORS.get(layer, 'gray'),
                    label=layer, linewidth=2, markersize=5)
        ax.set_title(type_names[t], color=type_colors[t], fontweight='bold')
        ax.set_xlabel('Recording Day')
        ax.set_ylabel('% cells' if t == 0 else '')
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(day_labels, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


# ── Statistical tests ─────────────────────────────────────────

def run_chi_square_tests(label_tensor, cell_layers, layer_list,
                          day_labels, n_types):
    print("\n── Chi-square: layer × cell type, per session ──")
    n_sessions = label_tensor.shape[1]
    for col, day in enumerate(day_labels):
        table = np.zeros((len(layer_list), n_types), dtype=int)
        for li, layer in enumerate(layer_list):
            if layer not in cell_layers:
                continue
            rows       = cell_layers[layer]
            col_labels = label_tensor[rows, col]
            valid      = col_labels >= 0
            for t in range(n_types):
                table[li, t] = int(np.sum(col_labels[valid] == t))
        if table.sum() == 0 or table.min() < 1:
            print(f"  {day}: sparse/no data — skipping")
            continue
        try:
            chi2, p, dof, _ = chi2_contingency(table)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            print(f"  {day}: chi2={chi2:.2f}  df={dof}  p={p:.4f}  {sig}")
        except Exception as e:
            print(f"  {day}: failed ({e})")


def run_friedman_tests(label_tensor, cell_layers, layer_list,
                        session_days, day_labels, type_names):
    """
    Friedman test per (cell type, layer): does cell-type membership change
    significantly across sessions within a layer?
    Uses only cells present in all sessions.
    """
    print("\n── Friedman test: change across sessions, per cell type × layer ──")
    x = np.array(session_days)

    for t, tname in enumerate(type_names):
        for layer in layer_list:
            if layer not in cell_layers:
                continue
            rows = cell_layers[layer]

            # Binary membership matrix: 1 if cell is type t in that session
            binary = np.zeros((len(rows), len(day_labels)))
            for i, row in enumerate(rows):
                for col in range(len(day_labels)):
                    lbl = label_tensor[row, col]
                    if lbl >= 0:
                        binary[i, col] = float(lbl == t)
                    else:
                        binary[i, col] = np.nan

            # Keep only cells present in all sessions
            complete = ~np.any(np.isnan(binary), axis=1)
            n_complete = int(np.sum(complete))
            if n_complete < 5:
                print(f"  {tname} × {layer}: n={n_complete} complete cells — skip")
                continue

            data = binary[complete, :]
            try:
                stat, p = friedmanchisquare(*[data[:, col]
                                               for col in range(len(day_labels))])
                sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                print(f"  {tname} × {layer}: Friedman chi2={stat:.2f}  p={p:.4f}  {sig}"
                      f"  (n={n_complete} complete cells)")
            except Exception as e:
                print(f"  {tname} × {layer}: failed ({e})")


def run_slope_tests(label_tensor, cell_layers, layer_list,
                     session_days, type_names):
    """
    Kruskal-Wallis test: do per-cell experience slopes differ across layers?
    For each tracked cell, fits a regression of binary type-membership
    (is this cell type t in session s?) across sessions → slope = rate of change.
    KW tests whether slope distributions differ between layers.
    Also plots violin distributions of slopes.
    """
    print("\n── Kruskal-Wallis: per-cell slopes across layers, per cell type ──")
    x = np.array(session_days, dtype=float)

    all_slopes = {}   # {(tname, layer): array of slopes}

    for t, tname in enumerate(type_names):
        slopes_by_layer = {}
        for layer in layer_list:
            if layer not in cell_layers:
                continue
            rows   = cell_layers[layer]
            slopes = []
            for row in rows:
                col_labels = label_tensor[row, :]
                valid      = col_labels >= 0
                if valid.sum() < 3:
                    continue
                y        = (col_labels[valid] == t).astype(float)
                sl, *_   = linregress(x[valid], y)
                slopes.append(sl)
            if slopes:
                slopes_by_layer[layer] = np.array(slopes)

        groups = [slopes_by_layer[l] for l in layer_list if l in slopes_by_layer]
        if len(groups) < 2:
            print(f"  {tname}: not enough layers")
            continue

        stat, p = kruskal(*groups)
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        print(f"  {tname}: KW H={stat:.2f}  p={p:.4f}  {sig}")
        for layer in layer_list:
            if layer in slopes_by_layer:
                s = slopes_by_layer[layer]
                print(f"    {layer}: n={len(s)}  median={np.median(s):+.4f}/day")

        all_slopes[tname] = slopes_by_layer

    return all_slopes


def plot_slope_violins(all_slopes, layer_list, type_names, type_colors,
                        animal_id, output_path=None):
    """Violin plot of per-cell slopes by layer for each cell type."""
    n_types = len(type_names)
    fig, axes = plt.subplots(1, n_types, figsize=(4.5 * n_types, 4), sharey=False)
    fig.suptitle(f'{animal_id} — Per-cell Experience Slopes by Layer (tracked)',
                 fontweight='bold')
    if n_types == 1:
        axes = [axes]

    for t, (tname, ax) in enumerate(zip(type_names, axes)):
        if tname not in all_slopes:
            ax.set_title(tname)
            continue

        data_layers = [all_slopes[tname].get(l, np.array([])) for l in layer_list]
        data_valid  = [(i, d) for i, d in enumerate(data_layers) if len(d) >= 3]

        if not data_valid:
            ax.set_title(tname)
            continue

        positions = [i for i, _ in data_valid]
        parts = ax.violinplot([d for _, d in data_valid],
                               positions=positions,
                               showmedians=True, showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor(type_colors[t])
            pc.set_alpha(0.6)
        for part in ['cmedians', 'cbars', 'cmaxes', 'cmins']:
            if part in parts:
                parts[part].set_color(type_colors[t])

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels([layer_list[i] for i in positions], fontsize=9)
        ax.set_title(tname, color=type_colors[t], fontweight='bold')
        ax.set_ylabel('Slope (Δ membership / day)' if t == 0 else '')
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


# ── Main ─────────────────────────────────────────────────────

if __name__ == '__main__':
    out_dir = OUTPUT_DIR or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Fit population model ───────────────────────────────
    print("\n=== 1. Fitting population PCA + k-means model ===")
    profiles_pop, bin_centers, landmark_positions = load_pca_data_for_model(PCA_DATA_FILE)

    pca, kmeans, optimal_k, raw_labels_pop, centroids = fit_model(
        profiles_pop, N_PCA_COMPONENTS, N_CLUSTER_PCS, K_RANGE, OVERRIDE_K
    )
    n_types = optimal_k
    type_names, type_colors, mean_profiles = assign_semantic_labels(
        raw_labels_pop, profiles_pop, n_types, bin_centers, landmark_positions
    )

    print("  Cell types:")
    for t in range(n_types):
        n   = int(np.sum(raw_labels_pop == t))
        print(f"    [{t}] {type_names[t]:25s}  n={n} ({n/len(raw_labels_pop)*100:.1f}%)")

    animal_id = animal_id_from_path(ROI_TRACKING_FILE)

    # ── 2. Load tracking ──────────────────────────────────────
    print("\n=== 2. Loading tracked ROIs ===")
    tracked_matrix, day_labels, session_dirs = load_tracking(ROI_TRACKING_FILE)
    session_days = parse_day_numbers(day_labels)
    n_sessions   = len(day_labels)

    if session_dirs:
        preproc_files = find_files_from_tracking(session_dirs, "*_preproc*.h5")
        smi_files     = find_files_from_tracking(session_dirs, "*_smi_results.h5")
    else:
        preproc_files = find_preproc_files(ANIMAL_DIR)
        smi_files     = find_smi_files(ANIMAL_DIR)

    report_found_files("Preproc", preproc_files, day_labels)
    report_found_files("SMI",     smi_files,     day_labels)

    # Layer assignment from reference session
    cell_layers = assign_layers_from_smi(
        tracked_matrix, day_labels, smi_files, REFERENCE_DAY
    )
    layer_list = [l for l in LAYER_ORDER
                  if l in cell_layers and len(cell_layers[l]) > 0]
    for layer in layer_list:
        print(f"  {layer}: {len(cell_layers[layer])} tracked cells")

    # ── 3. Build label tensor ─────────────────────────────────
    print("\n=== 3. Labelling tracked cells per session ===")
    label_tensor = build_label_tensor(
        tracked_matrix, day_labels, preproc_files,
        pca, kmeans, n_types,
        TRIM_START_CM, TRIM_END_CM, TARGET_N_BINS, SMOOTH_SIGMA
    )

    total_labelled = int(np.sum(label_tensor >= 0))
    print(f"\n  Total labelled cell-session pairs: {total_labelled} "
          f"/ {tracked_matrix.shape[0] * n_sessions}")

    # ── 4. Heatmap of label tensor ────────────────────────────
    print("\n=== 4. Label tensor heatmap ===")
    plot_label_tensor_heatmap(
        label_tensor, cell_layers, layer_list, day_labels,
        type_names, type_colors, animal_id,
        output_path=os.path.join(out_dir, f'{animal_id}_tracked_label_heatmap.png')
    )
    plt.show()

    # ── 5. Proportion matrix + plots ─────────────────────────
    print("\n=== 5. Layer proportion analysis ===")
    prop, counts = build_prop_matrix(
        label_tensor, cell_layers, layer_list, n_types
    )

    plot_stacked_bars_by_layer(
        prop, counts, layer_list, day_labels,
        type_names, type_colors, animal_id,
        output_path=os.path.join(out_dir, f'{animal_id}_tracked_stacked_bars.png')
    )
    plt.show()

    run_chi_square_tests(
        label_tensor, cell_layers, layer_list, day_labels, n_types
    )

    # ── 6. Experience trajectories ────────────────────────────
    print("\n=== 6. Experience trajectories ===")
    plot_trajectories_by_layer(
        prop, layer_list, session_days, day_labels,
        type_names, type_colors, animal_id,
        output_path=os.path.join(out_dir, f'{animal_id}_tracked_trajectories_by_layer.png')
    )
    plt.show()

    plot_trajectories_by_type(
        prop, layer_list, session_days, day_labels,
        type_names, type_colors, animal_id,
        output_path=os.path.join(out_dir, f'{animal_id}_tracked_trajectories_by_type.png')
    )
    plt.show()

    # ── 7. Per-cell statistics ────────────────────────────────
    print("\n=== 7. Statistical tests ===")
    run_friedman_tests(
        label_tensor, cell_layers, layer_list,
        session_days, day_labels, type_names
    )

    all_slopes = run_slope_tests(
        label_tensor, cell_layers, layer_list, session_days, type_names
    )

    plot_slope_violins(
        all_slopes, layer_list, type_names, type_colors, animal_id,
        output_path=os.path.join(out_dir, f'{animal_id}_tracked_slope_violins.png')
    )
    plt.show()

    print(f"\n=== Done — figures saved to: {out_dir} ===")
