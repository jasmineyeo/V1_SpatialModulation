"""
PCA_TrackedCells_AcrossAnimals.py
==================================
Population-level PCA + k-means on tracked cells pooled across animals.

Mirrors PCA_TrackedCells.py but loops over multiple animals, each with its
own tracking file, reference day, and analysis days.  Sessions are normalised
to a 1-based session index so trajectories align across animals.

Each row of the PCA matrix = one (tracked-cell × session × animal) observation.

Output figures saved to OUTPUT_DIR:
  tracked_across_scree.png
  tracked_across_k_selection.png
  tracked_across_pc_scatter.png
  tracked_across_mean_profiles.png
  tracked_across_global_proportions.png
  tracked_across_stacked_bars_by_layer.png
  tracked_across_trajectories_by_layer.png
  tracked_across_trajectories_by_type.png
  tracked_across_experience_regression.png
  tracked_across_session_trend.png

Prerequisite: TrackROIs.ipynb → roi_tracking_results.h5 per animal
              *_preproc*.h5 and *_smi_results.h5 per session per animal

JSY, 2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d as scipy_interp1d
from scipy.stats import kruskal, mannwhitneyu, linregress
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from load_tracked import (
    load_tracking, filter_to_analysis_days, find_files_from_tracking,
    find_preproc_files, find_smi_files, assign_layers_from_smi,
    load_preproc_session, build_reliability_mask, parse_day_numbers,
    animal_id_from_path, LAYER_ORDER, LAYER_COLORS, report_found_files,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Per-animal config: tracking file, reference day, analysis days
ANIMAL_CONFIGS = {
    'JSY052': {
        'roi_tracking_file': r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\TrackedROIs\roi_tracking_results.h5",
        'animal_dir':        r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging",
        'reference_day':     'Day2',
        'analysis_days':     ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7'],
    },
    'JSY054': {
        'roi_tracking_file': r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\TrackedROIs\roi_tracking_results.h5",
        'animal_dir':        r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging",
        'reference_day':     'Day2',
        'analysis_days':     ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7'],
    },
    'JSY055': {
        'roi_tracking_file': r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\TrackedROIs\roi_tracking_results.h5",
        'animal_dir':        r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging",
        'reference_day':     'Day1',
        'analysis_days':     ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7'],
    },
}

OUTPUT_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\tracked_pca_across_animals"

# Profile preprocessing — must match PCA_DataAggregation.py
TRIM_START_CM      = 10.0
TRIM_END_CM        = 120.0
TARGET_N_BINS      = 115
SMOOTH_SIGMA       = 1.0
EXCLUDE_FIRST_BINS = 10
EXCLUDE_LAST_BINS  = 10

# Cell selection — must match PCA_DataAggregation.py
# 'reliable_cells'       — basic reliability from preproc
# 'combined_reliable'    — stricter: CC, Cohen's d, pattern correlation
# 'reliable_valid_cells' — combined_reliable + valid SMI geometry (needs SMI h5)
CELL_SELECTION = 'combined_reliable'

# Fixed-pool tracking: if True, reliability is evaluated only on each animal's
# reference day and that fixed set is followed across all sessions.
FIXED_POOL = True

# Alignment
USE_ALIGNED_PROFILES = False

# Onset-cell classification
POST_ONSET_START_CM  = 35.0
ONSET_R_THRESHOLD    = 0.3
ONSET_MAX_SHIFT_CM   = 15.0
CELL_TYPE_LANDMARK       = 'landmark'
CELL_TYPE_ONSET_ONLY     = 'onset_only'
CELL_TYPE_ONSET_LANDMARK = 'onset_landmark'
TEMPLATE_SIGMA_CM        = 8.0

# Landmark positions (must match folder 3)
LANDMARK_POSITIONS = [25, 55, 85, 115]

# PCA / clustering
N_PCA_COMPONENTS = 10
N_CLUSTER_PCS    = 5
K_RANGE          = range(2, 8)
OVERRIDE_K       = None   # set to int to force k, or None for auto

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ============================================================


# ── Preprocessing helpers (same as PCA_TrackedCells.py) ─────

def _zscore(profile):
    mu, sigma = np.mean(profile), np.std(profile)
    return (profile - mu) / sigma if sigma > 0 else profile - mu


def _preprocess_one(raw_profile, bc, trim_start, trim_end, target_n_bins, smooth_sigma):
    if smooth_sigma > 0:
        raw_profile = gaussian_filter1d(raw_profile.astype(float), sigma=smooth_sigma)
    start_idx = np.searchsorted(bc, trim_start)
    end_idx   = np.searchsorted(bc, trim_end)
    trimmed   = raw_profile[start_idx:end_idx]
    bc_trim   = bc[start_idx:end_idx]
    common_bc = np.linspace(trim_start, trim_end, target_n_bins)
    f = scipy_interp1d(bc_trim, trimmed, kind='linear',
                       bounds_error=False, fill_value='extrapolate')
    return _zscore(f(common_bc)), common_bc


def _is_valid_cell(mean_profile, bc, roi_idx, reliable_cells,
                   exclude_first_bins, exclude_last_bins):
    if roi_idx < 0 or roi_idx >= len(reliable_cells):
        return False
    if not reliable_cells[roi_idx]:
        return False
    if np.max(mean_profile) == 0:
        return False
    bin_spacing   = np.mean(np.diff(bc))
    onset_thresh  = bc[0]  + exclude_first_bins * bin_spacing
    reward_thresh = bc[-1] - exclude_last_bins  * bin_spacing
    peak_pos      = bc[np.argmax(mean_profile)]
    if peak_pos > reward_thresh or peak_pos < onset_thresh:
        return False
    return True



# ── Per-animal profile aggregation ──────────────────────────

def aggregate_one_animal(animal_id, cfg):
    """
    Load tracking + preproc + SMI for one animal.
    Returns lists (to be concatenated across animals):
      profiles, session_indices, layer_labels, animal_labels, tracked_ids,
      n_tracked_per_layer (dict), bin_centers
    """
    print(f"\n{'='*60}")
    print(f"  Animal: {animal_id}")
    print(f"{'='*60}")

    roi_file      = cfg['roi_tracking_file']
    animal_dir    = cfg['animal_dir']
    reference_day = cfg['reference_day']
    analysis_days = cfg['analysis_days']

    if not os.path.exists(roi_file):
        print(f"  Tracking file not found: {roi_file}")
        return None

    # Load tracking matrix
    tracked_matrix, day_labels, session_dirs = load_tracking(roi_file)
    tracked_matrix, day_labels, session_dirs = filter_to_analysis_days(
        tracked_matrix, day_labels, session_dirs, analysis_days)

    if len(day_labels) == 0:
        print("  No sessions after filtering — skipping")
        return None

    # Find preproc + SMI files
    if session_dirs:
        preproc_files = find_files_from_tracking(session_dirs, "*_preproc*.h5")
        smi_files     = find_files_from_tracking(session_dirs, "*_smi_results.h5")
    else:
        preproc_files = find_preproc_files(animal_dir)
        smi_files     = find_smi_files(animal_dir)

    report_found_files("Preproc", preproc_files, day_labels)

    # Layer assignment from reference day
    cell_layers = assign_layers_from_smi(tracked_matrix, day_labels,
                                          smi_files, reference_day)
    n_tracked_per_layer = {l: len(cell_layers.get(l, [])) for l in LAYER_ORDER}
    for l in LAYER_ORDER:
        print(f"  {l}: {n_tracked_per_layer[l]} tracked cells")

    # Build reverse map: tracked row → layer
    cell_to_layer = {}
    for layer, rows in cell_layers.items():
        for r in rows:
            cell_to_layer[int(r)] = layer

    n_tracked = tracked_matrix.shape[0]

    ref_day = reference_day if FIXED_POOL else None
    rel_mask = build_reliability_mask(tracked_matrix, day_labels, preproc_files,
                                      cell_selection=CELL_SELECTION,
                                      smi_files=smi_files,
                                      reference_day=ref_day)

    profiles_out       = []
    session_idx_out    = []
    layer_out          = []
    animal_out         = []
    tracked_ids_out    = []
    common_bc          = None

    for col, day in enumerate(day_labels):
        if day not in preproc_files:
            print(f"  [WARNING] No preproc for {day} — skipping")
            continue

        session_idx = col + 1  # 1-based session index

        _, nsa, bc, _, _, _, _ = load_preproc_session(preproc_files[day])
        n_cells_sess = nsa.shape[0]

        mean_profiles_sess = np.mean(nsa, axis=1)
        n_valid = 0

        for row in range(n_tracked):
            if not rel_mask[row, col]:
                continue

            roi_idx = int(tracked_matrix[row, col])
            if roi_idx < 0 or roi_idx >= n_cells_sess:
                continue

            raw_profile = mean_profiles_sess[roi_idx]
            if not _is_valid_cell(raw_profile, bc, roi_idx,
                                   np.ones(n_cells_sess, dtype=bool),
                                   EXCLUDE_FIRST_BINS, EXCLUDE_LAST_BINS):
                continue

            proc, cbc = _preprocess_one(raw_profile, bc,
                                         TRIM_START_CM, TRIM_END_CM,
                                         TARGET_N_BINS, SMOOTH_SIGMA)
            if common_bc is None:
                common_bc = cbc

            profiles_out.append(proc)
            session_idx_out.append(session_idx)
            layer_out.append(cell_to_layer.get(row, 'Unknown'))
            animal_out.append(animal_id)
            tracked_ids_out.append(row)
            n_valid += 1

        print(f"  {day} (idx={session_idx}): {n_valid} valid observations")

    return {
        'profiles':            np.array(profiles_out),
        'session_indices':     np.array(session_idx_out, dtype=int),
        'layer_labels':        np.array(layer_out),
        'animal_labels':       np.array(animal_out),
        'tracked_ids':         np.array(tracked_ids_out, dtype=int),
        'n_tracked_per_layer': n_tracked_per_layer,
        'bin_centers':         common_bc,
    }


# ── PCA helpers (same as PCA_TrackedCells.py) ───────────────

def run_pca(profiles, n_components):
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(profiles)
    print(f"  Variance explained: "
          + ", ".join(f"PC{i+1}={v*100:.1f}%" for i, v in
                      enumerate(pca.explained_variance_ratio_[:5])))
    return pca, scores


def select_optimal_k(X, k_range):
    sil_scores, inertias = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil_scores.append(silhouette_score(X, labels))
        inertias.append(km.inertia_)
    auto_k = list(k_range)[int(np.argmax(sil_scores))]
    return sil_scores, inertias, auto_k


def fit_kmeans(X, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return km, labels


def assign_semantic_labels(raw_labels, profiles, n_types, bin_centers, landmark_positions):
    """Label clusters by peak position relative to landmarks."""
    mean_profiles = np.array([profiles[raw_labels == t].mean(axis=0)
                               for t in range(n_types)])
    peak_positions = np.array([bin_centers[np.argmax(mp)] for mp in mean_profiles])

    lm = np.array(landmark_positions)
    type_names  = []
    type_colors = []
    default_colors = ['#2ca02c', '#8c564b', '#e377c2', '#17becf',
                      '#ff7f0e', '#9467bd', '#7f7f7f']

    for t in range(n_types):
        pk = peak_positions[t]
        dists = np.abs(lm - pk)
        nearest_lm = int(np.argmin(dists))
        if dists[nearest_lm] < 20:
            name = f"LD{nearest_lm+1}-preferring"
        else:
            name = f"Intermediate spatial"
        type_names.append(name)
        type_colors.append(default_colors[t % len(default_colors)])

    return type_names, type_colors, mean_profiles


# ── Proportion matrix ────────────────────────────────────────

def build_prop_matrix(raw_labels, session_indices, layer_labels,
                      animal_labels, session_order, layer_list, n_types):
    """
    prop[layer][type][session_idx] = mean proportion across animals
    also returns per-animal proportions for stats
    """
    animals = sorted(set(animal_labels))
    # prop_per_animal[animal][layer][type][session_idx]
    prop_per_animal = {}
    for animal in animals:
        mask_a = animal_labels == animal
        prop_per_animal[animal] = {}
        for layer in layer_list:
            prop_per_animal[animal][layer] = np.full((n_types, len(session_order)), np.nan)
            for si, sess in enumerate(session_order):
                mask = mask_a & (session_indices == sess) & (layer_labels == layer)
                n = np.sum(mask)
                if n > 0:
                    for t in range(n_types):
                        prop_per_animal[animal][layer][t, si] = np.sum(raw_labels[mask] == t) / n

    # mean ± SEM across animals
    prop_mean = {}
    prop_sem  = {}
    for layer in layer_list:
        prop_mean[layer] = np.full((n_types, len(session_order)), np.nan)
        prop_sem[layer]  = np.full((n_types, len(session_order)), np.nan)
        for t in range(n_types):
            for si in range(len(session_order)):
                vals = [prop_per_animal[a][layer][t, si]
                        for a in animals
                        if not np.isnan(prop_per_animal[a][layer][t, si])]
                if vals:
                    prop_mean[layer][t, si] = np.mean(vals)
                    prop_sem[layer][t, si]  = np.std(vals) / np.sqrt(len(vals))

    return prop_mean, prop_sem, prop_per_animal


# ── Plotting ─────────────────────────────────────────────────

def plot_scree(pca, n_cluster_pcs, output_path=None):
    fig, ax = plt.subplots(figsize=(5, 3))
    evr = pca.explained_variance_ratio_
    ax.bar(range(1, len(evr)+1), evr * 100, color='steelblue', alpha=0.8)
    ax.axvline(n_cluster_pcs + 0.5, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('PC')
    ax.set_ylabel('Variance explained (%)')
    ax.set_title('Scree plot — pooled tracked cells (across animals)')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_mean_profiles(mean_profiles, bin_centers, landmark_positions,
                       type_names, type_colors, output_path=None):
    n_types = len(type_names)
    fig, axes = plt.subplots(1, n_types, figsize=(4*n_types, 3.5), sharey=True)
    fig.suptitle('Tracked PCA (across animals) — Mean profiles per cluster',
                 fontweight='bold')
    if n_types == 1:
        axes = [axes]
    for t, ax in enumerate(axes):
        ax.plot(bin_centers, mean_profiles[t], color='black', linewidth=2)
        for lp in landmark_positions:
            ax.axvline(lp, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_title(type_names[t], color=type_colors[t], fontweight='bold', fontsize=9)
        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Z-scored activity' if t == 0 else '')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_pc_scatter(pc_scores, raw_labels, type_names, type_colors,
                    animal_labels, output_path=None):
    animals = sorted(set(animal_labels))
    markers = ['o', '^', 's', 'D', 'v', 'P', '*']
    fig, ax = plt.subplots(figsize=(7, 6))
    for t, (name, color) in enumerate(zip(type_names, type_colors)):
        for ai, animal in enumerate(animals):
            mask = (raw_labels == t) & (animal_labels == animal)
            ax.scatter(pc_scores[mask, 0], pc_scores[mask, 1],
                       c=color, marker=markers[ai % len(markers)],
                       alpha=0.4, s=15, linewidths=0)
    # Legend — cluster colors
    cluster_patches = [mpatches.Patch(color=c, label=n)
                       for n, c in zip(type_names, type_colors)]
    animal_handles  = [plt.Line2D([0], [0], marker=markers[ai % len(markers)],
                                   color='gray', linestyle='', markersize=6,
                                   label=a)
                       for ai, a in enumerate(animals)]
    ax.legend(handles=cluster_patches, loc='upper left', fontsize=7,
              title='Cluster', title_fontsize=7)
    ax.add_artist(ax.legend(handles=animal_handles, loc='upper right',
                            fontsize=7, title='Animal', title_fontsize=7))
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('Tracked PCA (across animals) — PC scatter')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_trajectories_by_type(prop_mean, prop_sem, layer_list, session_order,
                               type_names, type_colors, output_path=None):
    n_types  = len(type_names)
    n_layers = len(layer_list)
    fig, axes = plt.subplots(n_types, n_layers,
                              figsize=(3.5*n_layers, 3*n_types), sharey='row',
                              sharex=True)
    if n_types == 1: axes = axes[np.newaxis, :]
    if n_layers == 1: axes = axes[:, np.newaxis]
    fig.suptitle('Tracked PCA (across animals) — Proportion trajectory per cluster × layer',
                 fontweight='bold')
    x = np.array(session_order)
    for t, (name, color) in enumerate(zip(type_names, type_colors)):
        for li, layer in enumerate(layer_list):
            ax = axes[t, li]
            y    = prop_mean[layer][t]
            yerr = prop_sem[layer][t]
            valid = ~np.isnan(y)
            if np.sum(valid) > 1:
                ax.plot(x[valid], y[valid], color=color, marker='o', markersize=4)
                ax.fill_between(x[valid], (y-yerr)[valid], (y+yerr)[valid],
                                alpha=0.2, color=color)
            if t == 0:
                ax.set_title(layer, fontsize=9,
                             color=LAYER_COLORS.get(layer, 'black'), fontweight='bold')
            if li == 0:
                ax.set_ylabel(f'{name}\n% cells', fontsize=8, color=color)
            if t == n_types - 1:
                ax.set_xlabel('Session index')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_experience_regression(prop_per_animal, layer_list, session_order,
                                type_names, type_colors, output_path=None):
    """OLS slope per animal per cluster × layer, pooled scatter + fit line."""
    n_types  = len(type_names)
    n_layers = len(layer_list)
    animals  = sorted(prop_per_animal.keys())
    animal_colors = plt.cm.tab10(np.linspace(0, 1, len(animals)))

    fig, axes = plt.subplots(n_types, n_layers,
                              figsize=(3.5*n_layers, 3*n_types),
                              sharey='row', sharex=True)
    if n_types == 1: axes = axes[np.newaxis, :]
    if n_layers == 1: axes = axes[:, np.newaxis]
    fig.suptitle('Tracked PCA (across animals) — Experience effect (OLS per animal)',
                 fontweight='bold')

    x_all = np.array(session_order)
    for t, (name, color) in enumerate(zip(type_names, type_colors)):
        for li, layer in enumerate(layer_list):
            ax = axes[t, li]
            slopes = []
            for ai, animal in enumerate(animals):
                y = prop_per_animal[animal][layer][t]
                valid = ~np.isnan(y)
                if np.sum(valid) >= 2:
                    ax.scatter(x_all[valid], y[valid]*100,
                               color=animal_colors[ai], s=20, alpha=0.7, zorder=3)
                    slope, intercept, r, p, _ = linregress(x_all[valid], y[valid]*100)
                    slopes.append(slope)
                    x_fit = np.linspace(x_all[valid].min(), x_all[valid].max(), 50)
                    lw = 2 if p < 0.05 else 1
                    ls = '-' if p < 0.05 else '--'
                    ax.plot(x_fit, slope*x_fit + intercept,
                            color=animal_colors[ai], linewidth=lw, linestyle=ls)
            if t == 0:
                ax.set_title(layer, fontsize=9,
                             color=LAYER_COLORS.get(layer, 'black'), fontweight='bold')
            if li == 0:
                ax.set_ylabel(f'{name}\n% cells', fontsize=8, color=color)
            if t == n_types - 1:
                ax.set_xlabel('Session index')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.2)

    # Animal legend
    handles = [mpatches.Patch(color=animal_colors[ai], label=a)
               for ai, a in enumerate(animals)]
    fig.legend(handles=handles, loc='lower right', fontsize=7,
               title='Animal', ncol=len(animals))
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


# ── Main ─────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("PCA — TRACKED CELLS ACROSS ANIMALS")
    print("=" * 65)

    # ── 1. Aggregate profiles from all animals ─────────────────
    all_profiles       = []
    all_session_idx    = []
    all_layer_labels   = []
    all_animal_labels  = []
    all_tracked_ids    = []
    bin_centers        = None
    n_tracked_summary  = {}

    for animal_id, cfg in ANIMAL_CONFIGS.items():
        result = aggregate_one_animal(animal_id, cfg)
        if result is None or len(result['profiles']) == 0:
            print(f"  No data for {animal_id} — skipping")
            continue
        all_profiles.append(result['profiles'])
        all_session_idx.append(result['session_indices'])
        all_layer_labels.append(result['layer_labels'])
        all_animal_labels.append(result['animal_labels'])
        all_tracked_ids.append(result['tracked_ids'])
        n_tracked_summary[animal_id] = result['n_tracked_per_layer']
        if bin_centers is None:
            bin_centers = result['bin_centers']

    if len(all_profiles) == 0:
        print("No data loaded across any animal — check configs.")
        return

    profiles       = np.concatenate(all_profiles, axis=0)
    session_idx    = np.concatenate(all_session_idx)
    layer_labels   = np.concatenate(all_layer_labels)
    animal_labels  = np.concatenate(all_animal_labels)

    print(f"\nPooled: {len(profiles)} observations across "
          f"{len(set(animal_labels))} animals")
    for layer in LAYER_ORDER:
        n = np.sum(layer_labels == layer)
        print(f"  {layer}: {n} observations")

    session_order = sorted(set(session_idx.tolist()))
    layer_list    = [l for l in LAYER_ORDER if np.any(layer_labels == l)]

    # ── 2. PCA ────────────────────────────────────────────────
    print("\n[2] PCA...")
    pca, pc_scores = run_pca(profiles, N_PCA_COMPONENTS)

    # ── 3. K selection + clustering ───────────────────────────
    print("\n[3] K selection + clustering...")
    X = pc_scores[:, :N_CLUSTER_PCS]
    sil_scores, inertias, auto_k = select_optimal_k(X, K_RANGE)
    optimal_k = OVERRIDE_K if OVERRIDE_K is not None else auto_k
    print(f"  K: auto={auto_k}  used={optimal_k}")

    _, raw_labels = fit_kmeans(X, optimal_k)
    n_types = optimal_k

    # ── 4. Semantic labels ────────────────────────────────────
    print("\n[4] Semantic labels...")
    type_names, type_colors, mean_profiles = assign_semantic_labels(
        raw_labels, profiles, n_types, bin_centers, LANDMARK_POSITIONS)
    for t in range(n_types):
        n = int(np.sum(raw_labels == t))
        print(f"  [{t}] {type_names[t]:30s}  n={n} ({n/len(raw_labels)*100:.1f}%)")

    # ── 5. Proportion matrix ──────────────────────────────────
    print("\n[5] Building proportion matrices...")
    prop_mean, prop_sem, prop_per_animal = build_prop_matrix(
        raw_labels, session_idx, layer_labels, animal_labels,
        session_order, layer_list, n_types)

    # ── 6. Figures ────────────────────────────────────────────
    print("\n[6] Generating figures...")
    prefix = os.path.join(OUTPUT_DIR, "tracked_across")

    plot_scree(pca, N_CLUSTER_PCS,
               f"{prefix}_scree.png")

    plot_mean_profiles(mean_profiles, bin_centers, LANDMARK_POSITIONS,
                       type_names, type_colors,
                       f"{prefix}_mean_profiles.png")

    plot_pc_scatter(pc_scores, raw_labels, type_names, type_colors,
                    animal_labels,
                    f"{prefix}_pc_scatter.png")

    plot_trajectories_by_type(prop_mean, prop_sem, layer_list, session_order,
                               type_names, type_colors,
                               f"{prefix}_trajectories_by_type.png")

    plot_experience_regression(prop_per_animal, layer_list, session_order,
                                type_names, type_colors,
                                f"{prefix}_experience_regression.png")

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    plt.show()


if __name__ == '__main__':
    main()
