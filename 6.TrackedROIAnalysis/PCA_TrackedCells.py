"""
PCA_TrackedCells.py
===================
Population-level PCA + k-means analysis, restricted to **tracked cells only**.

Mirrors 4.PCA/PCA_ComprehensiveAnalysis.py exactly, but instead of loading
pre-aggregated data from *_pca_data.h5, it builds the feature matrix on the
fly from the tracked-ROI file + per-session preproc HDF5 files.

Each row of the PCA matrix is one (tracked-cell × session) observation.
Cells must be valid (roi index ≥ 0, reliable, not onset/reward responders)
in a given session to be included for that session.

Output figures:
  *_tracked_scree.png
  *_tracked_k_selection.png
  *_tracked_pc_scatter.png
  *_tracked_mean_profiles.png
  *_tracked_global_proportions.png
  *_tracked_stacked_bars_by_layer.png
  *_tracked_trajectories_by_layer.png
  *_tracked_trajectories_by_type.png

Prerequisite: run TrackROIs.ipynb → roi_tracking_results.h5
              and ensure *_preproc*.h5 files exist per session.

JSY, 2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d as scipy_interp1d
from scipy.stats import chi2_contingency, kruskal
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from load_tracked import (
    load_tracking, filter_to_analysis_days, find_files_from_tracking,
    find_preproc_files, assign_layers_from_smi, load_preproc_session,
    parse_day_numbers, animal_id_from_path,
    LAYER_ORDER, LAYER_COLORS, report_found_files,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROI_TRACKING_FILE = r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\TrackedROIs\roi_tracking_results.h5"
ANIMAL_DIR        = r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging"
REFERENCE_DAY     = "Day2"
ANALYSIS_DAYS     = ['Day1','Day2', 'Day3', 'Day4', 'Day5']

# Profile preprocessing — must match PCA_DataAggregation.py
TRIM_START_CM     = 10.0
TRIM_END_CM       = 125.0
TARGET_N_BINS     = 115
SMOOTH_SIGMA      = 1.0
EXCLUDE_FIRST_BINS = 5
EXCLUDE_LAST_BINS  = 5

# PCA / clustering — must match 4.PCA/PCA_ComprehensiveAnalysis.py
N_PCA_COMPONENTS  = 10
N_CLUSTER_PCS     = 5
K_RANGE           = range(2, 8)
OVERRIDE_K        = 3   # set to int to force k, or None for auto

OUTPUT_DIR = os.path.join(ANIMAL_DIR, "TrackedROIs")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ============================================================


# ── Profile preprocessing helpers ────────────────────────────

def _zscore(profile):
    mu, sigma = np.mean(profile), np.std(profile)
    return (profile - mu) / sigma if sigma > 0 else profile - mu


def _preprocess_one(raw_profile, bc,
                    trim_start, trim_end, target_n_bins, smooth_sigma):
    """
    Trim → interpolate → z-score a single mean spatial profile.
    Returns (n_bins,) array on linspace(trim_start, trim_end, target_n_bins).
    """
    if smooth_sigma > 0:
        raw_profile = gaussian_filter1d(raw_profile.astype(float), sigma=smooth_sigma)

    start_idx = np.searchsorted(bc, trim_start)
    end_idx   = np.searchsorted(bc, trim_end)
    trimmed   = raw_profile[start_idx:end_idx]
    bc_trim   = bc[start_idx:end_idx]

    common_bc = np.linspace(trim_start, trim_end, target_n_bins)
    f = scipy_interp1d(bc_trim, trimmed, kind='linear',
                       bounds_error=False, fill_value='extrapolate')
    interpolated = f(common_bc)
    return _zscore(interpolated), common_bc


def _is_valid_cell(mean_profile, bc, roi_idx, reliable_cells,
                   exclude_first_bins, exclude_last_bins):
    """
    Return True if roi_idx is valid for PCA:
      - roi_idx ≥ 0
      - cell is reliable
      - global peak not in onset or reward exclusion zone
    """
    if roi_idx < 0 or roi_idx >= len(reliable_cells):
        return False
    if not reliable_cells[roi_idx]:
        return False

    profile = mean_profile
    if np.max(profile) == 0:
        return False

    bin_spacing   = np.mean(np.diff(bc))
    onset_thresh  = bc[0]  + exclude_first_bins  * bin_spacing
    reward_thresh = bc[-1] - exclude_last_bins    * bin_spacing
    peak_pos      = bc[np.argmax(profile)]

    return onset_thresh <= peak_pos <= reward_thresh


# ── Data aggregation from tracked ROIs ───────────────────────

def aggregate_tracked_profiles(tracked_matrix, day_labels, preproc_files,
                                cell_layers,
                                trim_start, trim_end, target_n_bins,
                                smooth_sigma, exclude_first_bins, exclude_last_bins):
    """
    Build the PCA feature matrix from tracked cells.

    Each valid (tracked_cell × session) pair becomes one row.

    Returns
    -------
    profiles       : (n_obs, target_n_bins) float — z-scored trimmed profiles
    bin_centers    : (target_n_bins,) float
    session_labels : (n_obs,) str
    layer_labels   : (n_obs,) str
    tracked_ids    : (n_obs,) int — row index in tracked_matrix
    """
    n_tracked, n_sessions = tracked_matrix.shape

    # Reverse-map each tracked cell to its layer
    cell_to_layer = {}
    for layer, rows in cell_layers.items():
        for r in rows:
            cell_to_layer[int(r)] = layer

    profiles_list       = []
    session_labels_list = []
    layer_labels_list   = []
    tracked_ids_list    = []
    common_bc           = None

    for col, day in enumerate(day_labels):
        if day not in preproc_files:
            print(f"  [WARNING] No preproc file for {day} — skipping")
            continue

        print(f"  Aggregating {day}...", end=' ', flush=True)
        _, nsa, bc, rel, _, _, _ = load_preproc_session(preproc_files[day])

        mean_profiles_sess = np.mean(nsa, axis=1)   # (n_cells_sess, n_bins)
        n_valid_day = 0

        for row in range(n_tracked):
            roi_idx = int(tracked_matrix[row, col])
            raw_profile = mean_profiles_sess[roi_idx] if (
                0 <= roi_idx < len(mean_profiles_sess)) else None

            if raw_profile is None:
                continue
            if not _is_valid_cell(raw_profile, bc, roi_idx, rel,
                                  exclude_first_bins, exclude_last_bins):
                continue

            proc, cbc = _preprocess_one(raw_profile, bc,
                                        trim_start, trim_end,
                                        target_n_bins, smooth_sigma)
            if common_bc is None:
                common_bc = cbc

            profiles_list.append(proc)
            session_labels_list.append(day)
            layer_labels_list.append(cell_to_layer.get(row, 'Unknown'))
            tracked_ids_list.append(row)
            n_valid_day += 1

        print(f"{n_valid_day} valid observations")

    profiles       = np.array(profiles_list)
    session_labels = np.array(session_labels_list)
    layer_labels   = np.array(layer_labels_list)
    tracked_ids    = np.array(tracked_ids_list, dtype=int)

    print(f"\n  Total observations for PCA: {len(profiles)}")
    print(f"  Feature dims: {profiles.shape[1]} bins "
          f"({trim_start}–{trim_end} cm)")
    return profiles, common_bc, session_labels, layer_labels, tracked_ids


# ── PCA + clustering (identical to PCA_ComprehensiveAnalysis) ─

def run_pca(profiles, n_components):
    pca    = PCA(n_components=n_components)
    scores = pca.fit_transform(profiles)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    print(f"  Variance explained: "
          + "  ".join([f"PC{i+1}={pca.explained_variance_ratio_[i]*100:.1f}%"
                       for i in range(min(5, n_components))])
          + f"  |  top-{n_components} cumulative: {cumvar[-1]:.1f}%")
    return pca, scores


def select_optimal_k(X, k_range, random_state=42):
    sil_scores, inertias = [], []
    for k in k_range:
        km  = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        lbs = km.fit_predict(X)
        sil_scores.append(silhouette_score(X, lbs))
        inertias.append(km.inertia_)
    auto_k = list(k_range)[int(np.argmax(sil_scores))]
    return sil_scores, inertias, auto_k


def fit_kmeans(X, k, random_state=42):
    km     = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(X)
    return km, labels


def assign_semantic_labels(raw_labels, profiles, n_types,
                            bin_centers, landmark_positions):
    mean_profiles = np.array([
        np.mean(profiles[raw_labels == t], axis=0)
        if np.sum(raw_labels == t) > 0 else np.zeros(profiles.shape[1])
        for t in range(n_types)
    ])
    peak_bins = np.argmax(mean_profiles, axis=1)

    names, used = [None] * n_types, set()

    def _pick(order):
        for k in order:
            if k not in used:
                used.add(k)
                return k

    names[_pick(np.argsort(peak_bins))]       = 'Adaptation-like'
    names[_pick(np.argsort(peak_bins)[::-1])] = 'L4-preferring'
    ranges = mean_profiles.max(axis=1) - mean_profiles.min(axis=1)
    names[_pick(np.argsort(ranges))]           = 'Visually responsive'

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


def build_prop_matrix(raw_labels, session_labels, layer_labels,
                       session_order, layer_list, n_types):
    n_sessions = len(session_order)
    n_layers   = len(layer_list)
    prop   = np.full((n_types, n_layers, n_sessions), np.nan)
    counts = np.zeros((n_layers, n_sessions), dtype=int)

    for si, sess in enumerate(session_order):
        sess_mask = session_labels == sess
        for li, layer in enumerate(layer_list):
            mask = sess_mask & (layer_labels == layer)
            n    = int(np.sum(mask))
            if n == 0:
                continue
            counts[li, si] = n
            for t in range(n_types):
                prop[t, li, si] = np.sum(raw_labels[mask] == t) / n

    return prop, counts


# ── Plotting (identical to PCA_ComprehensiveAnalysis) ─────────

def plot_scree(pca, n_cluster_pcs, animal_id, output_path=None):
    var    = pca.explained_variance_ratio_ * 100
    cumvar = np.cumsum(var)
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle(f'{animal_id} (tracked) — PCA Scree', fontweight='bold')
    ax.bar(range(1, len(var) + 1), var, color='steelblue', alpha=0.8)
    ax.plot(range(1, len(var) + 1), cumvar, 'ko-', markersize=4)
    ax.axvline(n_cluster_pcs + 0.5, color='red', linestyle='--',
               label=f'Clustering uses top {n_cluster_pcs} PCs')
    ax.set_xlabel('PC'); ax.set_ylabel('Variance explained (%)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def plot_k_selection(sil_scores, inertias, k_range, optimal_k,
                     animal_id, output_path=None):
    ks = list(k_range)
    fig, ax1 = plt.subplots(figsize=(5, 4))
    fig.suptitle(f'{animal_id} (tracked) — K selection', fontweight='bold')
    ax2 = ax1.twinx()
    ax1.plot(ks, sil_scores, 'b-o', label='Silhouette', linewidth=2)
    ax2.plot(ks, inertias,   'r-s', label='Inertia',    linewidth=2)
    ax1.axvline(optimal_k, color='green', linestyle='--', label=f'k={optimal_k}')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Silhouette score', color='blue')
    ax2.set_ylabel('Inertia',          color='red')
    ax1.set_xticks(ks)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def plot_pc_scatter(pc_scores, raw_labels, type_names, type_colors,
                    animal_id, output_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(f'{animal_id} (tracked) — PC scatter', fontweight='bold')
    for t, (name, col) in enumerate(zip(type_names, type_colors)):
        m = raw_labels == t
        ax.scatter(pc_scores[m, 0], pc_scores[m, 1],
                   c=col, alpha=0.5, s=15, label=f'{name} (n={m.sum()})')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(fontsize=7, markerscale=2); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def plot_mean_profiles(mean_profiles, bin_centers, landmark_positions,
                        type_names, type_colors, animal_id, output_path=None):
    n_types = len(type_names)
    fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 3.5), sharey=True)
    fig.suptitle(f'{animal_id} (tracked) — Mean profiles per cluster',
                 fontweight='bold')
    if n_types == 1:
        axes = [axes]
    for t, ax in enumerate(axes):
        ax.plot(bin_centers, mean_profiles[t], color=type_colors[t], linewidth=2)
        for lp in landmark_positions:
            ax.axvline(lp, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_title(type_names[t], color=type_colors[t],
                     fontweight='bold', fontsize=9)
        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Z-scored activity' if t == 0 else '')
        ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def plot_global_proportions(raw_labels, layer_labels, layer_list,
                             type_names, type_colors, animal_id, output_path=None):
    n_types  = len(type_names)
    n_layers = len(layer_list)
    props, ns = np.zeros((n_types, n_layers)), np.zeros(n_layers, dtype=int)
    for li, layer in enumerate(layer_list):
        mask = layer_labels == layer
        n    = int(np.sum(mask))
        ns[li] = n
        for t in range(n_types):
            props[t, li] = np.sum(raw_labels[mask] == t) / n if n > 0 else 0

    fig, ax = plt.subplots(figsize=(max(4, n_layers * 1.8), 4))
    fig.suptitle(f'{animal_id} (tracked) — Cell type proportions by layer (pooled)',
                 fontweight='bold')
    xpos    = np.arange(n_layers)
    bottoms = np.zeros(n_layers)
    for t in range(n_types):
        ax.bar(xpos, props[t] * 100, bottom=bottoms,
               color=type_colors[t], label=type_names[t], alpha=0.85, width=0.7)
        bottoms += props[t] * 100
    for xi, n in enumerate(ns):
        ax.text(xi, 103, f'n={n}', ha='center', va='bottom', fontsize=7, rotation=45)
    ax.set_xticks(xpos)
    ax.set_xticklabels(layer_list)
    ax.set_ylim(0, 120); ax.set_ylabel('% observations')
    handles = [mpatches.Patch(color=type_colors[t], label=type_names[t])
               for t in range(n_types)]
    ax.legend(handles=handles, fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def plot_stacked_bars_by_layer(prop, counts, layer_list, session_order,
                                type_names, type_colors, animal_id,
                                output_path=None):
    n_layers, n_types = len(layer_list), len(type_names)
    fig, axes = plt.subplots(1, n_layers, figsize=(3.5 * n_layers, 4), sharey=True)
    fig.suptitle(f'{animal_id} (tracked) — Cell type proportions per layer × session',
                 fontweight='bold')
    if n_layers == 1:
        axes = [axes]
    xpos = np.arange(len(session_order))
    for li, (layer, ax) in enumerate(zip(layer_list, axes)):
        bottoms = np.zeros(len(session_order))
        for t in range(n_types):
            vals    = prop[t, li, :]
            heights = np.where(~np.isnan(vals), vals * 100, 0)
            ax.bar(xpos, heights, bottom=bottoms,
                   color=type_colors[t],
                   label=type_names[t] if li == 0 else '',
                   width=0.7, alpha=0.85)
            bottoms += heights
        for xi in range(len(session_order)):
            n = counts[li, xi]
            if n > 0:
                ax.text(xi, 103, f'n={n}', ha='center', va='bottom',
                        fontsize=6, rotation=45)
        ax.set_title(layer, color=LAYER_COLORS.get(layer, 'black'), fontweight='bold')
        ax.set_xticks(xpos)
        ax.set_xticklabels(session_order, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 120)
        ax.set_ylabel('% observations' if li == 0 else '')
        ax.grid(True, alpha=0.2, axis='y')
    handles = [mpatches.Patch(color=type_colors[t], label=type_names[t])
               for t in range(n_types)]
    axes[-1].legend(handles=handles, fontsize=7, loc='upper right')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def _session_days(session_order):
    return [int(s.replace('Day', '')) for s in session_order]


def plot_trajectories_by_layer(prop, layer_list, session_order,
                                type_names, type_colors, animal_id,
                                output_path=None):
    n_layers, n_types = len(layer_list), len(type_names)
    x = np.array(_session_days(session_order))
    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4), sharey=True)
    fig.suptitle(f'{animal_id} (tracked) — Cell type trajectories per layer',
                 fontweight='bold')
    if n_layers == 1:
        axes = [axes]
    for li, (layer, ax) in enumerate(zip(layer_list, axes)):
        for t in range(n_types):
            y     = prop[t, li, :] * 100
            valid = ~np.isnan(y)
            if valid.sum() < 2:
                continue
            ax.plot(x[valid], y[valid], 'o-', color=type_colors[t],
                    label=type_names[t], linewidth=2, markersize=5)
        ax.set_title(layer, color=LAYER_COLORS.get(layer, 'black'), fontweight='bold')
        ax.set_xlabel('Recording day')
        ax.set_ylabel('% observations' if li == 0 else '')
        ax.set_ylim(0, 100); ax.set_xticks(x)
        ax.set_xticklabels(session_order, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def plot_trajectories_by_type(prop, layer_list, session_order,
                               type_names, type_colors, animal_id,
                               output_path=None):
    n_types = len(type_names)
    x = np.array(_session_days(session_order))
    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4), sharey=True)
    fig.suptitle(f'{animal_id} (tracked) — Layer trajectories per cell type',
                 fontweight='bold')
    if n_types == 1:
        axes = [axes]
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
        ax.set_xlabel('Recording day')
        ax.set_ylabel('% observations' if t == 0 else '')
        ax.set_ylim(0, 100); ax.set_xticks(x)
        ax.set_xticklabels(session_order, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


# ── Statistical tests ─────────────────────────────────────────

def run_chi_square_tests(raw_labels, session_labels, layer_labels,
                          session_order, layer_list, n_types):
    print("\n── Chi-square: layer × cell type, per session ──")
    for sess in session_order:
        sess_mask = session_labels == sess
        table = np.zeros((len(layer_list), n_types), dtype=int)
        for li, layer in enumerate(layer_list):
            mask = sess_mask & (layer_labels == layer)
            for t in range(n_types):
                table[li, t] = int(np.sum(raw_labels[mask] == t))
        if table.sum() == 0 or table.min() < 1:
            print(f"  {sess}: sparse/no data — skipping")
            continue
        try:
            chi2, p, dof, _ = chi2_contingency(table)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            print(f"  {sess}: chi2={chi2:.2f}  df={dof}  p={p:.4f}  {sig}")
        except Exception as e:
            print(f"  {sess}: failed ({e})")


def run_kruskal_layer_tests(raw_labels, layer_labels, layer_list,
                             n_types, type_names):
    print("\n── Kruskal-Wallis: layer effect on cell type proportion ──")
    for t, tname in enumerate(type_names):
        groups = []
        for layer in layer_list:
            mask = layer_labels == layer
            if np.sum(mask) == 0:
                continue
            groups.append((raw_labels[mask] == t).astype(float))
        if len(groups) < 2:
            print(f"  {tname}: not enough layers")
            continue
        try:
            stat, p = kruskal(*groups)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            print(f"  {tname}: KW H={stat:.2f}  p={p:.4f}  {sig}")
            for layer in layer_list:
                mask = layer_labels == layer
                if np.sum(mask) == 0:
                    continue
                pct = np.mean(raw_labels[mask] == t) * 100
                print(f"    {layer}: {pct:.1f}%  (n={np.sum(mask)})")
        except Exception as e:
            print(f"  {tname}: failed ({e})")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("PCA — TRACKED CELLS ONLY")
    print("=" * 65)

    animal_id = animal_id_from_path(ROI_TRACKING_FILE)
    prefix    = os.path.join(OUTPUT_DIR, f"{animal_id}_tracked")

    # ── 1. Load tracking ──────────────────────────────────────
    print("\n[1] Loading tracking matrix...")
    tracked_matrix, day_labels, session_dirs = load_tracking(ROI_TRACKING_FILE)
    tracked_matrix, day_labels, session_dirs = filter_to_analysis_days(
        tracked_matrix, day_labels, session_dirs, ANALYSIS_DAYS)

    print("\n[2] Finding preproc files...")
    if session_dirs:
        preproc_files = find_files_from_tracking(session_dirs, "*_preproc*.h5")
    else:
        preproc_files = find_preproc_files(ANIMAL_DIR)
    report_found_files("Preproc files", preproc_files, day_labels)

    # ── 2. Layer assignment ───────────────────────────────────
    print("\n[3] Assigning cells to layers (from SMI reference day)...")
    from load_tracked import find_files_from_tracking as _fft, find_smi_files
    if session_dirs:
        smi_files = _fft(session_dirs, "*_smi_results.h5")
    else:
        smi_files = find_smi_files(ANIMAL_DIR)
    cell_layers = assign_layers_from_smi(tracked_matrix, day_labels, smi_files,
                                          REFERENCE_DAY)
    for l in LAYER_ORDER:
        print(f"  {l}: {len(cell_layers.get(l, []))} tracked cells")

    # ── 3. Aggregate profiles ─────────────────────────────────
    print("\n[4] Aggregating spatial profiles from tracked cells...")
    profiles, bin_centers, session_labels, layer_labels, tracked_ids = \
        aggregate_tracked_profiles(
            tracked_matrix, day_labels, preproc_files, cell_layers,
            TRIM_START_CM, TRIM_END_CM, TARGET_N_BINS, SMOOTH_SIGMA,
            EXCLUDE_FIRST_BINS, EXCLUDE_LAST_BINS,
        )

    if len(profiles) == 0:
        print("No valid observations — check preproc files and tracking.")
        return

    session_order = [d for d in day_labels if d in session_labels]
    layer_list    = [l for l in LAYER_ORDER if np.any(layer_labels == l)]

    # Quick summary
    print(f"\n  Observations per session:")
    for sess in session_order:
        print(f"    {sess}: {np.sum(session_labels == sess)}")
    print(f"  Observations per layer:")
    for layer in layer_list:
        print(f"    {layer}: {np.sum(layer_labels == layer)}")

    # Landmark positions for profile plots (loaded from preproc context)
    # Use the same defaults as PCA_DataAggregation.py
    landmark_positions = [25, 55, 85, 115]

    # ── 4. PCA ────────────────────────────────────────────────
    print("\n[5] PCA...")
    pca, pc_scores = run_pca(profiles, N_PCA_COMPONENTS)

    # ── 5. K selection + clustering ───────────────────────────
    print("\n[6] K selection + clustering...")
    X = pc_scores[:, :N_CLUSTER_PCS]
    sil_scores, inertias, auto_k = select_optimal_k(X, K_RANGE)
    optimal_k = OVERRIDE_K if OVERRIDE_K is not None else auto_k
    print(f"  K: auto={auto_k}  used={optimal_k}")

    kmeans, raw_labels = fit_kmeans(X, optimal_k)
    n_types = optimal_k

    # ── 6. Semantic labels ────────────────────────────────────
    print("\n[7] Semantic labels...")
    type_names, type_colors, mean_profiles = assign_semantic_labels(
        raw_labels, profiles, n_types, bin_centers, landmark_positions)
    for t in range(n_types):
        n = int(np.sum(raw_labels == t))
        print(f"  [{t}] {type_names[t]:25s}  n={n} ({n/len(raw_labels)*100:.1f}%)")

    # ── 7. Figures ────────────────────────────────────────────
    print("\n[8] Generating figures...")

    plot_scree(pca, N_CLUSTER_PCS, animal_id,
               f"{prefix}_scree.png")

    plot_k_selection(sil_scores, inertias, K_RANGE, optimal_k, animal_id,
                     f"{prefix}_k_selection.png")

    plot_pc_scatter(pc_scores, raw_labels, type_names, type_colors, animal_id,
                    f"{prefix}_pc_scatter.png")

    plot_mean_profiles(mean_profiles, bin_centers, landmark_positions,
                       type_names, type_colors, animal_id,
                       f"{prefix}_mean_profiles.png")

    plot_global_proportions(raw_labels, layer_labels, layer_list,
                             type_names, type_colors, animal_id,
                             f"{prefix}_global_proportions.png")

    # ── 8. Layer × session proportion figures ─────────────────
    print("\n[9] Layer × session figures...")
    prop, counts = build_prop_matrix(
        raw_labels, session_labels, layer_labels,
        session_order, layer_list, n_types)

    plot_stacked_bars_by_layer(
        prop, counts, layer_list, session_order,
        type_names, type_colors, animal_id,
        f"{prefix}_stacked_bars_by_layer.png")

    plot_trajectories_by_layer(
        prop, layer_list, session_order,
        type_names, type_colors, animal_id,
        f"{prefix}_trajectories_by_layer.png")

    plot_trajectories_by_type(
        prop, layer_list, session_order,
        type_names, type_colors, animal_id,
        f"{prefix}_trajectories_by_type.png")

    # ── 9. Statistical tests ──────────────────────────────────
    print("\n[10] Statistical tests...")
    run_chi_square_tests(raw_labels, session_labels, layer_labels,
                          session_order, layer_list, n_types)
    run_kruskal_layer_tests(raw_labels, layer_labels, layer_list,
                             n_types, type_names)

    plt.show()
    print(f"\nDone. Outputs in: {OUTPUT_DIR}")

    return dict(profiles=profiles, bin_centers=bin_centers,
                session_labels=session_labels, layer_labels=layer_labels,
                tracked_ids=tracked_ids, raw_labels=raw_labels,
                type_names=type_names, pca=pca, kmeans=kmeans)


if __name__ == "__main__":
    main()
