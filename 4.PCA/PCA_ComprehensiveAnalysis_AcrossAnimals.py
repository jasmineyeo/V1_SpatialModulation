"""
PCA_ComprehensiveAnalysis_AcrossAnimals.py
==========================================
Population-level PCA + k-means functional cell type analysis across all animals.

Pools z-scored spatial profiles from all animals (each produced by
PCA_DataAggregation.py), fits a SINGLE PCA + k-means model on the pooled data,
and generates a full set of analysis figures.

This resolves the cluster-definition inconsistency that arises when animals
are clustered independently (different k values, heuristic label matching).
With a single model, every cell — regardless of animal — is assigned to
the same cluster definitions.

Steps:
  1.  Load *_pca_data.h5 for each animal
  2.  Pool all z-scored spatial profiles
  3.  Single PCA + k-means on the pooled data
  4.  Assign semantic labels to clusters
  5.  Population-level figures (scree, k-selection, PC scatter, mean profiles)
  6.  Per-animal proportion figures (by layer, by session trajectory)
  7.  Cross-animal summary figures (pooled layer proportions)
  8.  Statistical tests: layer effect + experience effect,
      with Fisher's method to combine p-values across animals
  9.  Statistical figures
  10. Save cluster labels back to each animal's HDF5

Prerequisite: run PCA_DataAggregation.py for each animal first.

JSY, 2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
from scipy.stats import (chi2_contingency, kruskal, mannwhitneyu,
                         linregress, combine_pvalues)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# CONFIGURATION
# ============================================================

ANIMALS = {
    'JSY040': r"D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\PCA\JSY040_pca_data.h5",  # excluded: outlier Visually responsive proportions
    # 'JSY041': r"D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\PCA\JSY041_pca_data.h5",  # excluded: outlier Visually responsive proportions

    # 'JSY044': r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\PCA\JSY044_pca_data.h5",  # excluded: inverted layer gradient
    'JSY052': r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\PCA\JSY052_pca_data.h5",  # excluded: outlier Visually responsive proportions
    'JSY051': r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\PCA\JSY051_pca_data.h5",
    'JSY054': r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\PCA\JSY054_pca_data.h5",
    'JSY055': r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\PCA\JSY055_pca_data.h5",
}

N_PCA_COMPONENTS = 10     # total PCs to fit
N_CLUSTER_PCS    = 5      # top PCs used for k-means
K_RANGE          = range(2, 8)
OVERRIDE_K       = None   # set to int to force a specific k

USE_ALIGNED_PROFILES =  False   # True  → use spatial_profiles_aligned (type-aware shifted)
                               # False → use spatial_profiles_zscore  (unshifted)

OUTPUT_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\across_animals_PCA_testing_knone_WO4144"

# ============================================================

LAYER_ORDER  = ['L2/3', 'L4', 'L5', 'L6']
LAYER_COLORS = {'L2/3': '#4CAF50', 'L4': '#2196F3', 'L5': '#FF9800', 'L6': '#9C27B0'}

# All animals plotted in black; distinguished by marker shape only
ANIMAL_COLORS = {
    'JSY044': 'black',
    'JSY051': 'black',
    'JSY052': 'black',
    'JSY054': 'black',
    'JSY055': 'black',
}
ANIMAL_MARKERS = {
    'JSY044': 'o',
    'JSY051': 's',
    'JSY052': '^',
    'JSY054': 'D',
    'JSY055': 'v',
}


# ── Data loading ──────────────────────────────────────────────

def load_animal(animal_id, filepath):
    """
    Load profiles + metadata from a single *_pca_data.h5.
    Returns a dict with profiles, labels, session order, bin info.
    """
    with h5py.File(filepath, 'r') as f:
        if USE_ALIGNED_PROFILES and 'features/spatial_profiles_aligned' in f:
            profiles = f['features/spatial_profiles_aligned'][:]
            print(f"  {animal_id}: using type-aware aligned profiles")
        elif not USE_ALIGNED_PROFILES and 'features/spatial_profiles_session_corrected' in f:
            profiles = f['features/spatial_profiles_session_corrected'][:]
            print(f"  {animal_id}: using session-corrected profiles")
        else:
            profiles = f['features/spatial_profiles_zscore'][:]
            print(f"  {animal_id}: using z-scored profiles (unshifted)")

        bin_centers        = f['metadata/bin_centers_trimmed'][:]
        landmark_positions = f['metadata/landmark_positions'][:]

        session_labels  = f['cells/session_labels'][:].astype(str)
        layer_labels    = f['cells/layer_labels'][:].astype(str)
        raw_session_ids = f['metadata/session_ids'][:].astype(str)

    session_order = sorted(raw_session_ids.tolist(),
                           key=lambda s: int(s.replace('Day', '')))
    print(f"  {animal_id}: {len(profiles)} cells  |  {len(session_order)} sessions")
    return {
        'animal_id':         animal_id,
        'profiles':          profiles,
        'session_labels':    session_labels,
        'layer_labels':      layer_labels,
        'session_order':     session_order,
        'bin_centers':       bin_centers,
        'landmark_positions': landmark_positions,
    }


def load_all_animals(animals_dict):
    """
    Load and pool data from all animals.
    Returns a pooled dict plus per-animal dicts.
    """
    animal_data = {}
    all_profiles, all_animal, all_session, all_layer = [], [], [], []
    bin_centers = landmark_positions = None

    for animal_id, filepath in animals_dict.items():
        if not os.path.exists(filepath):
            print(f"  WARNING: {animal_id} — file not found, skipping")
            continue
        data = load_animal(animal_id, filepath)
        animal_data[animal_id] = data

        n = len(data['profiles'])
        all_profiles.append(data['profiles'])
        all_animal.extend([animal_id] * n)
        all_session.extend(data['session_labels'].tolist())
        all_layer.extend(data['layer_labels'].tolist())

        if bin_centers is None:
            bin_centers        = data['bin_centers']
            landmark_positions = data['landmark_positions']

    pooled = {
        'profiles':           np.concatenate(all_profiles, axis=0),
        'animal_labels':      np.array(all_animal),
        'session_labels':     np.array(all_session),
        'layer_labels':       np.array(all_layer),
        'bin_centers':        bin_centers,
        'landmark_positions': landmark_positions,
        'animal_ids':         list(animal_data.keys()),
        'animal_data':        animal_data,
    }
    print(f"\n  Pooled: {len(pooled['profiles'])} cells "
          f"from {len(animal_data)} animals")
    return pooled


# ── PCA + clustering ──────────────────────────────────────────

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

    # 1. Earliest peak → Early-responding/Adaptation-like
    names[_pick(np.argsort(peak_bins))]       = 'Early-responding/Adaptation-like'
    # 2. Latest peak → LD4-preferring (reward landmark)
    names[_pick(np.argsort(peak_bins)[::-1])] = 'LD4-preferring'
    # 3. Among remaining, closest peak to LD1 landmark → LD1-preferring
    ld1_cm = landmark_positions[0] if len(landmark_positions) > 0 else 25.0
    peak_cms = np.array([
        bin_centers[peak_bins[k]] if peak_bins[k] < len(bin_centers) else np.inf
        for k in range(n_types)
    ])
    remaining = [k for k in range(n_types) if k not in used]
    if remaining:
        ld1_closest = remaining[int(np.argmin([abs(peak_cms[k] - ld1_cm)
                                               for k in remaining]))]
        names[_pick([ld1_closest])] = 'LD1-preferring'
    # 4. Any remaining cluster → Intermediate spatial
    for k in range(n_types):
        if names[k] is None:
            names[k] = 'Intermediate spatial'

    # Cluster colours chosen to avoid layer palette (green, blue, orange, purple)
    base_colors = {
        'Early-responding/Adaptation-like': '#AD1457',  # rose/magenta
        'LD4-preferring':                   '#00838F',  # teal
        'LD1-preferring':                   '#558B2F',  # olive green
        'Intermediate spatial':             '#6D4C41',  # brown
    }
    extra = ['#E65100', '#1565C0', '#4A148C', '#00695C']
    ec, colors = 0, []
    for name in names:
        colors.append(base_colors.get(name, extra[ec % len(extra)]))
        if name not in base_colors:
            ec += 1

    return names, colors, mean_profiles


# ── Proportion helpers ────────────────────────────────────────

def build_prop_matrix(raw_labels, session_labels, layer_labels,
                       session_order, layer_list, n_types):
    """(n_types, n_layers, n_sessions) proportion + count matrices."""
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


# ── Population-level figures ──────────────────────────────────

def plot_scree(pca, n_cluster_pcs, output_path=None):
    var    = pca.explained_variance_ratio_ * 100
    cumvar = np.cumsum(var)
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle('Pooled PCA — Scree', fontweight='bold')
    ax.bar(range(1, len(var) + 1), var, color='steelblue', alpha=0.8)
    ax.plot(range(1, len(var) + 1), cumvar, 'ko-', markersize=4)
    ax.axvline(n_cluster_pcs + 0.5, color='red', linestyle='--',
               label=f'Clustering uses top {n_cluster_pcs} PCs')
    ax.set_xlabel('PC')
    ax.set_ylabel('Variance explained (%)')
    ax.legend(fontsize=23)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_k_selection(sil_scores, inertias, k_range, optimal_k, output_path=None):
    ks  = list(k_range)
    fig, ax1 = plt.subplots(figsize=(5, 4))
    fig.suptitle('Pooled PCA — K selection', fontweight='bold')
    ax2 = ax1.twinx()
    ax1.plot(ks, sil_scores, 'b-o', label='Silhouette', linewidth=2)
    ax2.plot(ks, inertias,   'r-s', label='Inertia',    linewidth=2)
    ax1.axvline(optimal_k, color='green', linestyle='--', label=f'k={optimal_k}')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Silhouette score', color='blue')
    ax2.set_ylabel('Inertia',          color='red')
    ax1.set_xticks(ks)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=8)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_pc_scatter(pc_scores, raw_labels, animal_labels,
                    type_names, type_colors, animal_ids, output_path=None):
    """PC1 vs PC2 scatter: colour = cluster, marker = animal."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('Pooled PCA — PC scatter (colour=cluster, marker=animal)',
                 fontweight='bold')
    for t, (name, col) in enumerate(zip(type_names, type_colors)):
        for animal in animal_ids:
            m = (raw_labels == t) & (animal_labels == animal)
            ax.scatter(pc_scores[m, 0], pc_scores[m, 1],
                       c=col, marker=ANIMAL_MARKERS.get(animal, 'o'),
                       alpha=0.4, s=12,
                       label=f'{name} / {animal}' if t == 0 else '')
    # Cluster legend
    cluster_handles = [mpatches.Patch(color=type_colors[t], label=type_names[t])
                       for t in range(len(type_names))]
    # Animal legend
    animal_handles  = [plt.Line2D([0], [0], marker=ANIMAL_MARKERS.get(animal, 'o'),
                                  color='gray', linestyle='None', markersize=6,
                                  label=animal)
                       for animal in animal_ids]
    leg1 = ax.legend(handles=cluster_handles, fontsize=12,
                     loc='upper left',  title='Cluster')
    ax.add_artist(leg1)
    ax.legend(handles=animal_handles,  fontsize=12,
              loc='upper right', title='Animal')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_mean_profiles(mean_profiles, bin_centers, landmark_positions,
                        type_names, type_colors,
                        profiles=None, raw_labels=None, output_path=None):
    """Individual ROI traces (light) + mean (black, bold) per cluster."""
    n_types = len(type_names)
    fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 3.5), sharey=True)
    fig.suptitle('Pooled PCA — Mean profiles per cluster', fontweight='bold')
    if n_types == 1:
        axes = [axes]
    for t, ax in enumerate(axes):
        if profiles is not None and raw_labels is not None:
            for row in profiles[raw_labels == t]:
                ax.plot(bin_centers, row, color=type_colors[t],
                        alpha=0.08, linewidth=0.4)
        ax.plot(bin_centers, mean_profiles[t], color='black', linewidth=2, zorder=3)
        for lp in landmark_positions:
            ax.axvline(lp, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        n = int(np.sum(raw_labels == t)) if raw_labels is not None else ''
        ax.set_title(f'{type_names[t]} (n={n})',
                     color=type_colors[t], fontweight='bold', fontsize=9)
        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Z-scored activity' if t == 0 else '')
        ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


# ── Per-animal proportion figures ────────────────────────────

def plot_per_animal_stacked_bars(pooled, raw_labels, layer_list,
                                  type_names, type_colors, output_path=None):
    """
    One row per animal: stacked bar per layer (all sessions pooled).
    """
    animal_ids = pooled['animal_ids']
    n_animals  = len(animal_ids)
    n_layers   = len(layer_list)
    n_types    = len(type_names)

    fig, axes = plt.subplots(1, n_animals,
                             figsize=(max(4, n_layers * 1.6) * n_animals, 4),
                             sharey=True)
    fig.suptitle('Cell type proportions per layer — per animal (sessions pooled)',
                 fontweight='bold')
    if n_animals == 1:
        axes = [axes]

    xpos = np.arange(n_layers)
    for ai, animal in enumerate(animal_ids):
        ax   = axes[ai]
        mask = pooled['animal_labels'] == animal
        al   = pooled['layer_labels'][mask]
        rl   = raw_labels[mask]

        bottoms = np.zeros(n_layers)
        for t in range(n_types):
            heights = []
            for layer in layer_list:
                lmask = al == layer
                n     = int(np.sum(lmask))
                heights.append(np.mean(rl[lmask] == t) * 100 if n > 0 else 0)
            ax.bar(xpos, heights, bottom=bottoms,
                   color=type_colors[t],
                   label=type_names[t] if ai == 0 else '',
                   alpha=0.85, width=0.7)
            bottoms += np.array(heights)

        for xi, layer in enumerate(layer_list):
            n = int(np.sum(al == layer))
            ax.text(xi, 103, f'n={n}', ha='center', va='bottom',
                    fontsize=7, rotation=45)

        ax.set_title(animal, fontweight='bold')
        ax.set_xticks(xpos)
        ax.set_xticklabels(layer_list)
        ax.set_ylim(0, 120)
        ax.set_ylabel('% cells' if ai == 0 else '')
        ax.grid(True, alpha=0.2, axis='y')

    handles = [mpatches.Patch(color=type_colors[t], label=type_names[t])
               for t in range(n_types)]
    axes[-1].legend(handles=handles, fontsize=8, loc='upper right')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_per_animal_trajectories(pooled, raw_labels, layer_list,
                                  type_names, type_colors, output_path_template=None):
    """
    One figure per animal: proportion trajectories per layer.
    output_path_template should contain {animal}, e.g. 'path/{animal}_traj.png'.
    """
    figs = {}
    for animal in pooled['animal_ids']:
        amask   = pooled['animal_labels'] == animal
        al      = pooled['layer_labels'][amask]
        sl      = pooled['session_labels'][amask]
        rl      = raw_labels[amask]
        so      = pooled['animal_data'][animal]['session_order']
        ll      = [l for l in layer_list if np.any(al == l)]

        prop, _ = build_prop_matrix(rl, sl, al, so, ll, len(type_names))
        days    = np.array([int(s.replace('Day', '')) for s in so], dtype=float)

        n_types_local = len(type_names)
        n_layers      = len(ll)
        fig, axes = plt.subplots(1, n_layers,
                                 figsize=(4.5 * n_layers, 4), sharey=True)
        fig.suptitle(f'{animal} — Cell type trajectories per layer',
                     fontweight='bold')
        if n_layers == 1:
            axes = [axes]

        for li, (layer, ax) in enumerate(zip(ll, axes)):
            for t in range(n_types_local):
                y     = prop[t, li, :] * 100
                valid = ~np.isnan(y)
                if valid.sum() < 2:
                    continue
                ax.plot(days[valid], y[valid], 'o-',
                        color=type_colors[t], label=type_names[t],
                        linewidth=2, markersize=5)
            ax.set_title(layer,
                         color=LAYER_COLORS.get(layer, 'black'), fontweight='bold')
            ax.set_xlabel('Recording day')
            ax.set_ylabel('% cells' if li == 0 else '')
            ax.set_ylim(0, 100)
            ax.set_xticks(days)
            ax.set_xticklabels(so, rotation=45, ha='right', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2, axis='y')
        plt.tight_layout()
        figs[animal] = fig

        if output_path_template:
            path = output_path_template.format(animal=animal)
            fig.savefig(path, dpi=150, bbox_inches='tight')
    return figs


# ── Cross-animal summary figure ───────────────────────────────

def plot_adaptation_cells_diagnostic(profiles, bin_centers, landmark_positions,
                                      raw_labels, type_names,
                                      label='pooled', n_cells=50, cells_per_fig=10,
                                      output_dir=None):
    """
    For the Adaptation-like cluster: plot individual cell traces (original
    aligned profile) alongside what the profile would look like if the cell
    were force-shifted to L1.  10 cells per figure, 2 columns (original | shifted).
    Cells are sorted by peak position within each batch.
    """
    adapt_t = next((t for t, name in enumerate(type_names)
                    if 'adapt' in name.lower()), None)
    if adapt_t is None:
        print("  No Adaptation-like cluster found — skipping diagnostic")
        return []

    adapt_idx = np.where(raw_labels == adapt_t)[0]
    print(f"\n  Adaptation-like cluster: {len(adapt_idx)} cells  "
          f"(sampling {min(n_cells, len(adapt_idx))})")

    rng = np.random.default_rng(42)
    sample_size = min(n_cells, len(adapt_idx))
    sampled     = rng.choice(adapt_idx, size=sample_size, replace=False)
    # Sort by peak position so traces within each figure are ordered
    sampled     = sampled[np.argsort(np.argmax(profiles[sampled], axis=1))]

    l1_pos      = landmark_positions[0]
    bin_spacing = float(np.mean(np.diff(bin_centers)))
    n_figs      = int(np.ceil(sample_size / cells_per_fig))
    figs        = []

    for fig_idx in range(n_figs):
        batch  = sampled[fig_idx * cells_per_fig : (fig_idx + 1) * cells_per_fig]
        n_rows = len(batch)

        fig, axes = plt.subplots(n_rows, 2, figsize=(8, 2.2 * n_rows), sharex=True)
        fig.suptitle(
            f'{label} — Adaptation-like: original vs force-shifted to L1\n'
            f'(figure {fig_idx+1}/{n_figs},  '
            f'cells {fig_idx*cells_per_fig+1}–{fig_idx*cells_per_fig+n_rows}'
            f' of {sample_size})',
            fontweight='bold', fontsize=9)

        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for row, cell_i in enumerate(batch):
            orig    = profiles[cell_i]
            peak_cm = bin_centers[np.argmax(orig)]
            shift_bins = int(round((l1_pos - peak_cm) / bin_spacing))
            shifted = np.zeros_like(orig)
            if shift_bins > 0:
                shifted[shift_bins:] = orig[:-shift_bins]
            elif shift_bins < 0:
                shifted[:shift_bins] = orig[-shift_bins:]
            else:
                shifted = orig.copy()

            for col, (trace, col_title) in enumerate([
                (orig,    f'original  peak={peak_cm:.0f}cm'),
                (shifted, f'→ L1 ({l1_pos:.0f}cm)  shift={shift_bins:+d}bins'),
            ]):
                ax = axes[row, col]
                ax.plot(bin_centers, trace, color='#E53935', linewidth=1.2)
                ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
                for lp in landmark_positions:
                    ax.axvline(lp, color='green', linewidth=0.7,
                               linestyle='--', alpha=0.5)
                ax.set_ylabel(f'cell {cell_i}', fontsize=7, rotation=0,
                              labelpad=28, va='center')
                if row == 0:
                    ax.set_title(col_title, fontsize=8)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.15, axis='y')

        for ax in axes[-1]:
            ax.set_xlabel('Position (cm)', fontsize=8)

        plt.tight_layout()
        figs.append(fig)

        if output_dir:
            path = os.path.join(output_dir,
                                f'{label}_adapt_diagnostic_{fig_idx+1}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"    Saved: {os.path.basename(path)}")

    return figs


def plot_pooled_proportions_by_layer(pooled, raw_labels, layer_list,
                                      n_types, type_names, type_colors,
                                      output_path=None):
    """
    One panel per cell type: bar = mean proportion per layer (all animals pooled),
    dots = per-animal values coloured by animal.
    """
    animal_ids = pooled['animal_ids']
    fig, axes  = plt.subplots(1, n_types, figsize=(4 * n_types, 5))
    fig.suptitle('Cell type proportions by layer — pooled across animals',
                 fontweight='bold')
    if n_types == 1:
        axes = [axes]

    xpos = np.arange(len(layer_list))
    for t, (tname, tcolor, ax) in enumerate(zip(type_names, type_colors, axes)):
        # Grand-average bar
        grand_mean = []
        for layer in layer_list:
            lmask = pooled['layer_labels'] == layer
            n     = int(np.sum(lmask))
            grand_mean.append(np.mean(raw_labels[lmask] == t) * 100 if n > 0 else 0)
        ax.bar(xpos, grand_mean, color=tcolor, alpha=0.65, width=0.6, zorder=2)

        # Per-animal dots
        for animal in animal_ids:
            amask  = pooled['animal_labels'] == animal
            al     = pooled['layer_labels'][amask]
            rl     = raw_labels[amask]
            vals   = []
            for layer in layer_list:
                lm = al == layer
                n  = int(np.sum(lm))
                vals.append(np.mean(rl[lm] == t) * 100 if n > 0 else np.nan)
            ax.scatter(xpos, vals, color='black',
                       marker=ANIMAL_MARKERS.get(animal, 'o'),
                       s=60, zorder=3, label=animal, clip_on=False)

        ax.set_xticks(xpos)
        ax.set_xticklabels(layer_list, fontsize=9)
        ax.set_title(tname, color=tcolor, fontweight='bold')
        ax.set_ylabel('% cells' if t == 0 else '')
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.2, axis='y')
        if t == n_types - 1:
            animal_handles = [plt.Line2D([0], [0], color='black',
                                         marker=ANIMAL_MARKERS.get(a, 'o'),
                                         linestyle='None', markersize=6, label=a)
                              for a in animal_ids]
            ax.legend(handles=animal_handles, fontsize=7, loc='upper right')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


# ── Statistical helpers ───────────────────────────────────────

def _sig_stars(p):
    return '***' if p < 0.001 else ('**' if p < 0.01
           else ('*' if p < 0.05 else 'ns'))


def _draw_sig_bracket(ax, x1, x2, y, h, text, fontsize=8):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c='black')
    ax.text((x1 + x2) / 2, y + h, text,
            ha='center', va='bottom', fontsize=fontsize)


# ── Statistical tests ─────────────────────────────────────────

def run_layer_effect_fisher(pooled, raw_labels, layer_list, n_types, type_names):
    """
    Layer effect on cell type proportion.
    Per animal: Kruskal-Wallis. Across animals: Fisher's method.
    Returns results[tname] = {'per_animal': {animal: (kw_p, {layer: pct})},
                               'fisher_p': float}
    """
    print("\n── Layer effect (KW per animal + Fisher's combined) ──")
    animal_ids = pooled['animal_ids']
    results = {}

    for t, tname in enumerate(type_names):
        print(f"\n  {tname}:")
        per_animal_p = []
        per_animal   = {}

        for animal in animal_ids:
            amask  = pooled['animal_labels'] == animal
            al     = pooled['layer_labels'][amask]
            rl     = raw_labels[amask]

            groups = []
            for layer in layer_list:
                lm = al == layer
                if np.sum(lm) == 0:
                    continue
                groups.append((rl[lm] == t).astype(float))

            if len(groups) < 2:
                continue
            try:
                _, kw_p = kruskal(*groups)
                per_animal_p.append(kw_p)
                layer_pcts = {}
                for layer in layer_list:
                    lm = al == layer
                    if np.sum(lm) == 0:
                        continue
                    pct = np.mean(rl[lm] == t) * 100
                    layer_pcts[layer] = pct
                    print(f"      {layer}: {pct:.1f}%  (n={int(np.sum(lm))})")
                per_animal[animal] = (kw_p, layer_pcts)
                print(f"    {animal}: KW p={kw_p:.4f}  {_sig_stars(kw_p)}")
            except Exception as e:
                print(f"    {animal}: failed ({e})")

        fisher_p = np.nan
        if len(per_animal_p) >= 2:
            _, fisher_p = combine_pvalues(per_animal_p, method='fisher')
            print(f"    Fisher combined (n={len(per_animal_p)} animals): "
                  f"p={fisher_p:.4f}  {_sig_stars(fisher_p)}")
        results[tname] = {'per_animal': per_animal, 'fisher_p': fisher_p}

    return results


def run_experience_effect_fisher(pooled, raw_labels, layer_list, type_names):
    """
    Experience effect: proportion ~ recording day.
    Per animal (collapsed across layers) + per animal × layer: OLS regression.
    Across animals: Fisher's method per cell type (collapsed) and per cell type × layer.
    Returns results[tname]['all_layers'] and results[tname][layer] = {
        'animals': [...], 'slopes': [...], 'p_vals': [...],
        'fisher_p': float, 'consistent': bool, 'mean_slope': float }
    """
    print("\n── Experience effect (OLS per animal × layer + Fisher's combined) ──")
    animal_ids = pooled['animal_ids']
    results = {}

    for t, tname in enumerate(type_names):
        print(f"\n  {tname}:")
        results[tname] = {}

        # ── Collapsed across all layers ───────────────────────────────────────
        print("    all layers (collapsed):")
        per_animal_p, per_animal_dir, slopes_out, animals_out = [], [], [], []

        for animal in animal_ids:
            amask = pooled['animal_labels'] == animal
            sl    = pooled['session_labels'][amask]
            rl    = raw_labels[amask]
            so    = pooled['animal_data'][animal]['session_order']
            days  = np.array([int(s.replace('Day', '')) for s in so], dtype=float)

            props, valid_days = [], []
            for si, sess in enumerate(so):
                mask = (sl == sess)
                n    = int(np.sum(mask))
                if n == 0:
                    continue
                props.append(np.mean(rl[mask] == t) * 100)
                valid_days.append(days[si])

            if len(valid_days) < 3:
                continue
            try:
                slope, _, _, p, _ = linregress(np.array(valid_days),
                                               np.array(props))
                per_animal_p.append(p)
                per_animal_dir.append(1 if slope > 0 else -1)
                slopes_out.append(slope)
                animals_out.append(animal)
                print(f"      {animal}: slope={slope:+.2f}%/day  "
                      f"p={p:.4f}  {_sig_stars(p)}")
            except Exception as e:
                print(f"      {animal}: failed ({e})")

        fisher_p   = np.nan
        consistent = False
        if len(per_animal_p) >= 2:
            _, fisher_p = combine_pvalues(per_animal_p, method='fisher')
            consistent  = len(set(per_animal_dir)) == 1
            dir_str = ('all increase' if all(d > 0 for d in per_animal_dir)
                       else 'all decrease' if all(d < 0 for d in per_animal_dir)
                       else 'inconsistent direction')
            print(f"      Fisher combined: p={fisher_p:.4f}  "
                  f"{_sig_stars(fisher_p)}  [{dir_str}]"
                  f"{'  ✓' if consistent else '  ✗'}")

        results[tname]['all_layers'] = {
            'animals':    animals_out,
            'slopes':     slopes_out,
            'p_vals':     per_animal_p,
            'fisher_p':   fisher_p,
            'consistent': consistent,
            'mean_slope': float(np.mean(slopes_out)) if slopes_out else np.nan,
        }

        # ── Per layer ─────────────────────────────────────────────────────────
        for layer in layer_list:
            print(f"    {layer}:")
            per_animal_p, per_animal_dir, slopes_out, animals_out = [], [], [], []

            for animal in animal_ids:
                amask = pooled['animal_labels'] == animal
                al    = pooled['layer_labels'][amask]
                sl    = pooled['session_labels'][amask]
                rl    = raw_labels[amask]
                so    = pooled['animal_data'][animal]['session_order']
                days  = np.array([int(s.replace('Day', '')) for s in so],
                                 dtype=float)

                props, valid_days = [], []
                for si, sess in enumerate(so):
                    mask = (sl == sess) & (al == layer)
                    n    = int(np.sum(mask))
                    if n == 0:
                        continue
                    props.append(np.mean(rl[mask] == t) * 100)
                    valid_days.append(days[si])

                if len(valid_days) < 3:
                    continue
                try:
                    slope, _, _, p, _ = linregress(np.array(valid_days),
                                                   np.array(props))
                    per_animal_p.append(p)
                    per_animal_dir.append(1 if slope > 0 else -1)
                    slopes_out.append(slope)
                    animals_out.append(animal)
                    print(f"      {animal}: slope={slope:+.2f}%/day  "
                          f"p={p:.4f}  {_sig_stars(p)}")
                except Exception as e:
                    print(f"      {animal}: failed ({e})")

            fisher_p  = np.nan
            consistent = False
            if len(per_animal_p) >= 2:
                _, fisher_p = combine_pvalues(per_animal_p, method='fisher')
                consistent  = len(set(per_animal_dir)) == 1
                dir_str = ('all increase' if all(d > 0 for d in per_animal_dir)
                           else 'all decrease' if all(d < 0 for d in per_animal_dir)
                           else 'inconsistent direction')
                print(f"      Fisher combined: p={fisher_p:.4f}  "
                      f"{_sig_stars(fisher_p)}  [{dir_str}]"
                      f"{'  ✓' if consistent else '  ✗'}")

            results[tname][layer] = {
                'animals':    animals_out,
                'slopes':     slopes_out,
                'p_vals':     per_animal_p,
                'fisher_p':   fisher_p,
                'consistent': consistent,
                'mean_slope': float(np.mean(slopes_out)) if slopes_out else np.nan,
            }


    return results


def run_session_trend_tests_fisher(pooled, raw_labels, layer_list, type_names,
                                    n_early=2, n_late=2):
    """
    Per cell type × layer × animal:
      1. KW across sessions — omnibus "did anything change?"
      2. Early vs late MW-U — net direction.
    Across animals: Fisher's method on KW p-values.

    Returns results[tname][layer] = {
        'per_animal': {animal: {'kw_p', 'mw_p', 'direction',
                                'early_mean', 'late_mean', 'days', 'props'}},
        'fisher_p': float, 'consistent_direction': bool,
        'dominant_direction': str }
    """
    print(f"\n── Session trend: KW (omnibus) + early({n_early}) vs "
          f"late({n_late}) MW-U + Fisher's combined ──")
    animal_ids = pooled['animal_ids']
    results = {}

    for t, tname in enumerate(type_names):
        print(f"\n  {tname}:")
        results[tname] = {}
        for layer in layer_list:
            print(f"    {layer}:")
            per_animal_kw_p, directions = [], []
            per_animal = {}

            for animal in animal_ids:
                amask = pooled['animal_labels'] == animal
                al    = pooled['layer_labels'][amask]
                sl    = pooled['session_labels'][amask]
                rl    = raw_labels[amask]
                so    = pooled['animal_data'][animal]['session_order']
                days  = np.array([int(s.replace('Day', '')) for s in so],
                                 dtype=float)

                # Per-session proportions
                props, valid_days = [], []
                for si, sess in enumerate(so):
                    mask = (sl == sess) & (al == layer)
                    n    = int(np.sum(mask))
                    if n == 0:
                        continue
                    props.append(np.mean(rl[mask] == t) * 100)
                    valid_days.append(days[si])

                if len(props) < 3:
                    continue

                props      = np.array(props)
                valid_days = np.array(valid_days)

                # KW across sessions
                kw_groups = []
                for sess in so:
                    mask = (sl == sess) & (al == layer)
                    if np.sum(mask) == 0:
                        continue
                    kw_groups.append((rl[mask] == t).astype(float))

                kw_p = np.nan
                if len(kw_groups) >= 2:
                    try:
                        _, kw_p = kruskal(*kw_groups)
                    except Exception:
                        pass

                # Early vs late MW-U
                early_vals = props[:n_early]
                late_vals  = props[len(props) - n_late:]
                mw_p, direction = np.nan, 'n/a'
                if len(early_vals) >= 1 and len(late_vals) >= 1:
                    try:
                        _, mw_p = mannwhitneyu(early_vals, late_vals,
                                               alternative='two-sided')
                        direction = ('increase'
                                     if np.mean(late_vals) > np.mean(early_vals)
                                     else 'decrease')
                    except Exception:
                        pass

                print(f"      {animal}: KW p={kw_p:.4f} {_sig_stars(kw_p)}  |  "
                      f"early={np.mean(early_vals):.1f}% → "
                      f"late={np.mean(late_vals):.1f}%  "
                      f"MW p={mw_p:.4f} {_sig_stars(mw_p)}  [{direction}]")

                if not np.isnan(kw_p):
                    per_animal_kw_p.append(kw_p)
                    directions.append(direction)
                per_animal[animal] = {
                    'kw_p': kw_p, 'mw_p': mw_p, 'direction': direction,
                    'early_mean': float(np.mean(early_vals)),
                    'late_mean':  float(np.mean(late_vals)),
                    'days': valid_days, 'props': props,
                }

            fisher_p = np.nan
            consistent = False
            dominant   = 'n/a'
            if len(per_animal_kw_p) >= 2:
                _, fisher_p = combine_pvalues(per_animal_kw_p, method='fisher')
                consistent  = len(set(directions)) == 1
                n_inc = directions.count('increase')
                n_dec = directions.count('decrease')
                dominant = ('increase' if n_inc > n_dec else
                            'decrease' if n_dec > n_inc else 'mixed')
                print(f"      Fisher combined: p={fisher_p:.4f} "
                      f"{_sig_stars(fisher_p)}  [{dominant}]"
                      f"{'  ✓' if consistent else '  ✗'}")

            results[tname][layer] = {
                'per_animal':          per_animal,
                'fisher_p':            fisher_p,
                'consistent_direction': consistent,
                'dominant_direction':  dominant,
            }

    return results


def plot_session_trend_across_animals(trend_results, pooled, layer_list,
                                       type_names, type_colors,
                                       n_early=2, n_late=2,
                                       output_path=None):
    """
    Grid: rows = cell types, columns = layers.
    Each panel: per-animal proportion trajectories (coloured by animal),
    with early/late shading and Fisher KW p + dominant direction annotated.
    """
    animal_ids = pooled['animal_ids']
    n_types    = len(type_names)
    n_layers   = len(layer_list)

    fig, axes = plt.subplots(n_types, n_layers,
                             figsize=(4 * n_layers, 3.5 * n_types),
                             sharey='row', sharex='col')
    fig.suptitle('Session trend: KW omnibus + early vs late (per animal)',
                 fontweight='bold')

    if n_types == 1:
        axes = axes[np.newaxis, :]
    if n_layers == 1:
        axes = axes[:, np.newaxis]

    for t, tname in enumerate(type_names):
        for li, layer in enumerate(layer_list):
            ax  = axes[t, li]
            res = trend_results[tname].get(layer, {})

            if t == 0:
                ax.set_title(layer,
                             color=LAYER_COLORS.get(layer, 'black'),
                             fontweight='bold')
            if li == 0:
                ax.set_ylabel(f'{tname}\n% cells',
                              color=type_colors[t], fontsize=8)

            per_animal = res.get('per_animal', {})
            if not per_animal:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=7, color='gray')
                ax.set_xlabel('Session index' if t == n_types - 1 else '')
                ax.grid(True, alpha=0.2)
                continue

            # Find max n_sessions for x-axis
            max_sess = max(len(v['props']) for v in per_animal.values()
                           if v is not None)

            # Shade early / late
            ax.axvspan(-0.5, n_early - 0.5,
                       color='skyblue', alpha=0.12, label='early')
            ax.axvspan(max_sess - n_late - 0.5, max_sess - 0.5,
                       color='salmon', alpha=0.12, label='late')

            for animal, ares in per_animal.items():
                if ares is None:
                    continue
                xpos = np.arange(len(ares['props']))
                ax.plot(xpos, ares['props'],
                        color='black', linewidth=1.5, alpha=0.7,
                        marker=ANIMAL_MARKERS.get(animal, 'o'),
                        markersize=4, label=animal)

            fisher_p  = res.get('fisher_p', np.nan)
            dominant  = res.get('dominant_direction', 'n/a')
            consistent = res.get('consistent_direction', False)
            # kw_str  = (f"Fisher KW p={fisher_p:.3f} {_sig_stars(fisher_p)}"
            #            if not np.isnan(fisher_p) else "Fisher KW: n/a")
            # dir_str = f"{dominant}{'  ✓' if consistent else '  ✗'}"
            # ax.annotate(f"{kw_str}\n{dir_str}",
            #             xy=(0.03, 0.97), xycoords='axes fraction',
            #             fontsize=6.5, va='top', color='black')

            ax.set_xlabel('Session index' if t == n_types - 1 else '')
            ax.set_xticks(range(max_sess))
            ax.grid(True, alpha=0.2)

    # Animal marker legend on last panel
    animal_handles = [plt.Line2D([0], [0], color='black',
                                 marker=ANIMAL_MARKERS.get(a, 'o'),
                                 linewidth=1.5, markersize=4, label=a)
                      for a in animal_ids]
    axes[0, -1].legend(handles=animal_handles, fontsize=6, loc='upper right')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


# ── Statistical figures ───────────────────────────────────────

def plot_layer_posthoc_pooled(pooled, raw_labels, layer_list, n_types,
                               type_names, type_colors, output_path=None):
    """
    Bar = grand-mean proportion per layer, per-animal dots,
    + Bonferroni-corrected Mann-Whitney U brackets on pooled data.
    KW omnibus p (pooled) in x-label.
    """
    fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 5))
    fig.suptitle('Layer effect — pooled data + post-hoc (Bonferroni)',
                 fontweight='bold')
    if n_types == 1:
        axes = [axes]

    xpos = np.arange(len(layer_list))
    for t, (tname, tcolor, ax) in enumerate(zip(type_names, type_colors, axes)):
        # Grand-mean bars
        grand_mean = []
        for layer in layer_list:
            lm = pooled['layer_labels'] == layer
            n  = int(np.sum(lm))
            grand_mean.append(np.mean(raw_labels[lm] == t) * 100 if n > 0 else 0)
        ax.bar(xpos, grand_mean, color=tcolor, alpha=0.65, width=0.6, zorder=2)

        # Per-animal dots
        for animal in pooled['animal_ids']:
            amask = pooled['animal_labels'] == animal
            al    = pooled['layer_labels'][amask]
            rl    = raw_labels[amask]
            vals  = []
            for layer in layer_list:
                lm = al == layer
                n  = int(np.sum(lm))
                vals.append(np.mean(rl[lm] == t) * 100 if n > 0 else np.nan)
            ax.scatter(xpos, vals, color='black',
                       marker=ANIMAL_MARKERS.get(animal, 'o'),
                       s=60, zorder=3, clip_on=False)

        # Pairwise Mann-Whitney U + Bonferroni on pooled data
        groups, grp_layers = [], []
        for layer in layer_list:
            lm = pooled['layer_labels'] == layer
            if np.sum(lm) == 0:
                continue
            groups.append((raw_labels[lm] == t).astype(float))
            grp_layers.append(layer)

        pairs   = [(i, j) for i in range(len(groups))
                           for j in range(i + 1, len(groups))]
        n_pairs = len(pairs)
        sig_pairs = []
        for i, j in pairs:
            try:
                _, p_mw = mannwhitneyu(groups[i], groups[j],
                                       alternative='two-sided')
                p_adj = min(p_mw * n_pairs, 1.0)
                if p_adj < 0.05:
                    xi = layer_list.index(grp_layers[i])
                    xj = layer_list.index(grp_layers[j])
                    sig_pairs.append((xi, xj, _sig_stars(p_adj)))
            except Exception:
                pass

        y_top = max(grand_mean) if grand_mean else 0
        step  = y_top * 0.13
        base  = y_top * 1.08
        sig_pairs_sorted = sorted(sig_pairs, key=lambda x: abs(x[1] - x[0]))
        placed = []
        for xi, xj, stars in sig_pairs_sorted:
            level = 0
            while any(not (xj < px or xi > pxj) and lv == level
                      for px, pxj, lv in placed):
                level += 1
            placed.append((xi, xj, level))
            _draw_sig_bracket(ax, xi, xj, base + level * step, step * 0.25, stars)

        ax.set_xticks(xpos)
        ax.set_xticklabels(layer_list, fontsize=9)
        ax.set_title(tname, color=tcolor, fontweight='bold')
        ax.set_ylabel('% cells' if t == 0 else '')
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.2, axis='y')
        try:
            _, kw_p = kruskal(*groups)
            ax.set_xlabel(f'KW p={kw_p:.4f} {_sig_stars(kw_p)} (pooled)',
                          fontsize=8)
        except Exception:
            pass

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_experience_regression_all_animals(pooled, raw_labels, layer_list,
                                            type_names, type_colors,
                                            output_path=None):
    """
    Grid: rows = animals, columns = cell types.
    Each panel: scatter + OLS regression per layer.
    Solid line = p<0.05, dashed = ns.
    """
    animal_ids = pooled['animal_ids']
    n_types    = len(type_names)
    n_animals  = len(animal_ids)

    fig, axes = plt.subplots(n_animals, n_types,
                             figsize=(5 * n_types, 4 * n_animals),
                             sharey='row', sharex='col')
    fig.suptitle('Experience effect — proportion ~ day (per animal × cell type)',
                 fontweight='bold')

    # Ensure 2-D axes array
    if n_animals == 1:
        axes = axes[np.newaxis, :]
    if n_types == 1:
        axes = axes[:, np.newaxis]

    for ai, animal in enumerate(animal_ids):
        amask = pooled['animal_labels'] == animal
        al    = pooled['layer_labels'][amask]
        sl    = pooled['session_labels'][amask]
        rl    = raw_labels[amask]
        so    = pooled['animal_data'][animal]['session_order']
        days  = np.array([int(s.replace('Day', '')) for s in so], dtype=float)

        for t, (tname, tcolor) in enumerate(zip(type_names, type_colors)):
            ax = axes[ai, t]
            for layer in layer_list:
                lcolor = LAYER_COLORS.get(layer, 'gray')
                props, valid_days = [], []
                for si, sess in enumerate(so):
                    mask = (sl == sess) & (al == layer)
                    n    = int(np.sum(mask))
                    if n == 0:
                        continue
                    props.append(np.mean(rl[mask] == t) * 100)
                    valid_days.append(days[si])

                if len(valid_days) < 3:
                    continue
                x  = np.array(valid_days)
                yv = np.array(props)
                slope, intercept, _, p, _ = linregress(x, yv)
                ax.scatter(x, yv, color=lcolor, s=30, zorder=3, alpha=0.85)
                x_line = np.linspace(x.min(), x.max(), 50)
                ax.plot(x_line, slope * x_line + intercept, color=lcolor,
                        linewidth=2 if p < 0.05 else 1,
                        linestyle='-' if p < 0.05 else '--',
                        label=f'{layer} {slope:+.1f}%/d {_sig_stars(p)}')

            if t == 0:
                ax.set_ylabel(f'{animal}\n% cells', fontsize=9)
            if ai == 0:
                ax.set_title(tname, color=tcolor, fontweight='bold')
            ax.set_xticks(days)
            ax.set_xticklabels(so if ai == n_animals - 1 else [],
                               rotation=45, ha='right', fontsize=7)
            ax.legend(fontsize=6, loc='best')
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_layer_effect_summary(layer_results, type_names,
                               animal_ids, output_path=None):
    """
    Two-panel figure summarising the layer effect:
      Left  — KW p-value heatmap: rows=animals, cols=cell types.
              Colour = -log10(p), stars = significance.
      Right — Fisher combined p per cell type (bar chart).
    """
    n_types   = len(type_names)
    n_animals = len(animal_ids)

    fig, axes = plt.subplots(1, 2, figsize=(4 + n_types * 1.2, max(4, n_animals * 0.9 + 1.5)))
    fig.suptitle('Layer effect summary (KW per animal + Fisher combined)',
                 fontweight='bold')

    # Left: KW p heatmap (animals × cell types)
    kw_matrix = np.full((n_animals, n_types), np.nan)
    for t, tname in enumerate(type_names):
        for ai, animal in enumerate(animal_ids):
            entry = layer_results[tname]['per_animal'].get(animal)
            if entry is not None:
                kw_matrix[ai, t] = entry[0]   # kw_p

    ax = axes[0]
    log_p = -np.log10(np.clip(kw_matrix, 1e-10, 1))
    im = ax.imshow(log_p, aspect='auto', cmap='Reds', vmin=0, vmax=4)
    ax.set_xticks(range(n_types))
    ax.set_xticklabels(type_names, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(n_animals))
    ax.set_yticklabels(animal_ids, fontsize=8)
    ax.set_title('KW p-value per animal\n(darker = more significant)', fontsize=9)
    for ai in range(n_animals):
        for t in range(n_types):
            p = kw_matrix[ai, t]
            if not np.isnan(p):
                ax.text(t, ai, _sig_stars(p), ha='center', va='center',
                        fontsize=9, color='white' if p < 0.01 else 'black')
    plt.colorbar(im, ax=ax, label='-log10(p)')

    # Right: Fisher combined p per cell type
    ax = axes[1]
    fisher_ps = [layer_results[tname]['fisher_p'] for tname in type_names]
    colors_bar = ['#E53935' if p < 0.001 else '#FF9800' if p < 0.01
                  else '#FFC107' if p < 0.05 else '#9E9E9E'
                  for p in fisher_ps]
    xpos = np.arange(n_types)
    ax.bar(xpos, [-np.log10(max(p, 1e-10)) for p in fisher_ps],
           color=colors_bar, alpha=0.85)
    ax.axhline(-np.log10(0.05), color='black', linestyle='--',
               linewidth=0.8, label='p=0.05')
    ax.axhline(-np.log10(0.01), color='gray', linestyle=':',
               linewidth=0.8, label='p=0.01')
    for xi, p in enumerate(fisher_ps):
        ax.text(xi, -np.log10(max(p, 1e-10)) + 0.05, _sig_stars(p),
                ha='center', va='bottom', fontsize=9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(type_names, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('-log10(p)')
    ax.set_title("Fisher's combined p across animals", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_experience_summary_heatmap(exp_results, type_names, layer_list,
                                     animal_ids, output_path=None):
    """
    Two-panel figure summarising the experience effect:
      Left  — Mean slope heatmap: rows=cell types, cols=layers.
               Diverging colour (red=increase, blue=decrease).
               Stars = Fisher's combined p. Hatching = inconsistent direction.
      Right — Per-animal slope strip: each animal's slope per cell type × layer
               shown as dots, with the mean slope as a horizontal line.
    """
    n_types  = len(type_names)
    n_layers = len(layer_list)

    # Build matrices
    slope_matrix     = np.full((n_types, n_layers), np.nan)
    fisher_p_matrix  = np.full((n_types, n_layers), np.nan)
    consistent_matrix = np.zeros((n_types, n_layers), dtype=bool)

    for t, tname in enumerate(type_names):
        for li, layer in enumerate(layer_list):
            entry = exp_results[tname].get(layer, {})
            slope_matrix[t, li]      = entry.get('mean_slope', np.nan)
            fisher_p_matrix[t, li]   = entry.get('fisher_p',   np.nan)
            consistent_matrix[t, li] = entry.get('consistent', False)

    fig, axes = plt.subplots(1, 2, figsize=(5 + n_layers * 1.0, max(4, n_types * 1.0 + 2)))
    fig.suptitle('Experience effect summary (slope ~ day, Fisher combined)',
                 fontweight='bold')

    # Left: mean slope heatmap
    ax = axes[0]
    vmax = np.nanmax(np.abs(slope_matrix)) or 1
    im = ax.imshow(slope_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layer_list, fontsize=9)
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(type_names, fontsize=8)
    ax.set_title('Mean slope (%/day)\nstars = Fisher p, hatch = inconsistent', fontsize=8)
    for t in range(n_types):
        for li in range(n_layers):
            p  = fisher_p_matrix[t, li]
            ms = slope_matrix[t, li]
            if not np.isnan(ms):
                stars = _sig_stars(p) if not np.isnan(p) else ''
                ax.text(li, t, f'{ms:+.1f}\n{stars}',
                        ha='center', va='center', fontsize=7,
                        color='white' if abs(ms) > vmax * 0.6 else 'black')
            # Hatch inconsistent cells
            if not consistent_matrix[t, li]:
                ax.add_patch(plt.Rectangle((li - 0.5, t - 0.5), 1, 1,
                                           fill=False, hatch='///',
                                           edgecolor='gray', linewidth=0))
    plt.colorbar(im, ax=ax, label='Mean slope (%/day)')

    # Right: per-animal slopes per cell type × layer
    ax = axes[1]
    ax.set_title('Per-animal slopes', fontsize=9)
    ytick_labels, ytick_pos = [], []
    y_offset = 0
    gap = 0.4

    for t, tname in enumerate(type_names):
        for li, layer in enumerate(layer_list):
            entry   = exp_results[tname].get(layer, {})
            slopes  = entry.get('slopes', [])
            animals = entry.get('animals', [])
            mean_s  = entry.get('mean_slope', np.nan)
            y = y_offset

            for animal, slope in zip(animals, slopes):
                ax.scatter(slope, y, color='black',
                           marker=ANIMAL_MARKERS.get(animal, 'o'),
                           s=40, zorder=3, clip_on=False)
            if not np.isnan(mean_s):
                ax.plot([mean_s, mean_s], [y - 0.3, y + 0.3],
                        color='black', linewidth=2, zorder=4)

            ytick_labels.append(f'{tname[:6]}·{layer}')
            ytick_pos.append(y)
            y_offset += 1
        y_offset += gap

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels, fontsize=6)
    ax.set_xlabel('Slope (%/day)')
    ax.grid(True, alpha=0.2, axis='x')

    # Animal marker legend
    animal_handles = [plt.Line2D([0], [0], color='black',
                                  marker=ANIMAL_MARKERS.get(a, 'o'),
                                  linestyle='None', markersize=6, label=a)
                      for a in animal_ids]
    ax.legend(handles=animal_handles, fontsize=6, loc='lower right')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


# ── Cell-level statistical tests (pooled across animals) ─────

def run_layer_effect_pooled_cells(pooled, raw_labels, layer_list, n_types, type_names):
    """
    Layer effect tested at the cell level on ALL pooled cells.

    Approach: chi-squared on the contingency table [n_layers × n_types].
    Post-hoc: pairwise chi-squared between every pair of layers,
              Bonferroni-corrected for number of pairs.

    Returns results[tname] = {'chi2': float, 'p': float,
                               'posthoc': {(layerA, layerB): p_adj}}
    """
    print("\n── Layer effect — cell-level chi-squared (pooled) ──")
    results = {}

    # Full omnibus: one contingency table per cell type (layer × in/out)
    # Also build the overall [n_layers × n_types] table for a single omnibus
    n_layers = len(layer_list)
    contingency = np.zeros((n_layers, n_types), dtype=int)
    for li, layer in enumerate(layer_list):
        lm = pooled['layer_labels'] == layer
        for t in range(n_types):
            contingency[li, t] = int(np.sum(raw_labels[lm] == t))

    try:
        chi2_omni, p_omni, dof, _ = chi2_contingency(contingency)
        print(f"  Omnibus chi2={chi2_omni:.2f}  dof={dof}  "
              f"p={p_omni:.4e}  {_sig_stars(p_omni)}")
        print(f"  Total cells: {int(contingency.sum())}")
    except Exception as e:
        chi2_omni, p_omni = np.nan, np.nan
        print(f"  Omnibus failed: {e}")

    # Pairwise post-hoc per cell type
    n_pairs = n_layers * (n_layers - 1) // 2
    for t, tname in enumerate(type_names):
        print(f"\n  {tname}:")
        posthoc = {}
        for li in range(n_layers):
            for lj in range(li + 1, n_layers):
                la, lb = layer_list[li], layer_list[lj]
                lma = pooled['layer_labels'] == la
                lmb = pooled['layer_labels'] == lb
                na_in  = int(np.sum(raw_labels[lma] == t))
                na_out = int(np.sum(lma)) - na_in
                nb_in  = int(np.sum(raw_labels[lmb] == t))
                nb_out = int(np.sum(lmb)) - nb_in
                table  = np.array([[na_in, na_out], [nb_in, nb_out]])
                try:
                    _, p_raw, _, _ = chi2_contingency(table)
                    p_adj = min(p_raw * n_pairs, 1.0)
                    posthoc[(la, lb)] = p_adj
                    pct_a = na_in / max(na_in + na_out, 1) * 100
                    pct_b = nb_in / max(nb_in + nb_out, 1) * 100
                    print(f"    {la}({pct_a:.1f}%) vs {lb}({pct_b:.1f}%): "
                          f"p_adj={p_adj:.4f}  {_sig_stars(p_adj)}")
                except Exception:
                    posthoc[(la, lb)] = np.nan
        results[tname] = {
            'chi2_omni': chi2_omni, 'p_omni': p_omni,
            'posthoc': posthoc,
        }

    return results


def run_experience_effect_pooled_cells(pooled, raw_labels, layer_list, type_names):
    """
    Experience effect tested at the cell level on ALL pooled cells.

    For each (animal × session × layer) tuple, compute the proportion of
    each cell type. Pool all these data points and run a single OLS
    regression: proportion ~ recording_day.

    This gives n_animals × n_sessions data points per layer (vs. just
    n_animals in the Fisher's method), making the test well-powered.

    Returns results[tname][layer] = {
        'slope': float, 'p': float, 'r2': float, 'n_points': int,
        'days': list, 'props': list }
    """
    print("\n── Experience effect — cell-level OLS (pooled sessions) ──")
    results = {}

    for t, tname in enumerate(type_names):
        print(f"\n  {tname}:")
        results[tname] = {}
        for layer in layer_list:
            all_days, all_props = [], []

            for animal in pooled['animal_ids']:
                amask = pooled['animal_labels'] == animal
                al    = pooled['layer_labels'][amask]
                sl    = pooled['session_labels'][amask]
                rl    = raw_labels[amask]
                so    = pooled['animal_data'][animal]['session_order']

                for sess in so:
                    day = int(sess.replace('Day', ''))
                    mask = (sl == sess) & (al == layer)
                    n    = int(np.sum(mask))
                    if n == 0:
                        continue
                    prop = np.mean(rl[mask] == t) * 100
                    all_days.append(day)
                    all_props.append(prop)

            if len(all_days) < 3:
                results[tname][layer] = {
                    'slope': np.nan, 'p': np.nan,
                    'r2': np.nan, 'n_points': len(all_days),
                }
                continue

            x  = np.array(all_days,  dtype=float)
            y  = np.array(all_props, dtype=float)
            slope, intercept, r, p, _ = linregress(x, y)
            r2 = r ** 2
            results[tname][layer] = {
                'slope': slope, 'intercept': intercept, 'p': p, 'r2': r2,
                'n_points': len(x), 'days': x, 'props': y,
            }
            print(f"    {layer}: slope={slope:+.2f}%/day  "
                  f"r2={r2:.3f}  p={p:.4f}  {_sig_stars(p)}"
                  f"  (n={len(x)} session×animal points)")

    return results


def run_layer_effect_per_session(pooled, raw_labels, layer_list, n_types, type_names):
    """
    For each recording session (day), test whether cell type proportions
    differ across layers — at two levels:

    1. Cell-level (pooled): chi-squared on [n_layers × n_types] contingency
       table using every cell recorded on that day across all animals.
    2. Animal-level (Fisher's): KW per animal on that day, combined with
       Fisher's method.

    Sessions are matched by day number across animals (Day1, Day2, ...).

    Returns results[tname] = {
        day_num: {
            'p_cell':   float,  # chi-squared p on pooled cells
            'chi2':     float,
            'n_cells':  int,    # total cells that day
            'p_fisher': float,  # Fisher's combined p across animals
            'n_animals':int,    # animals contributing that day
        }
    }
    """
    print("\n── Layer effect per session (cell-level + Fisher's across animals) ──")

    # Collect all unique day numbers across animals
    all_days = set()
    for animal in pooled['animal_ids']:
        for sess in pooled['animal_data'][animal]['session_order']:
            all_days.add(int(sess.replace('Day', '')))
    all_days = sorted(all_days)

    results = {tname: {} for tname in type_names}

    for day in all_days:
        sess_label = f'Day{day}'
        day_mask   = pooled['session_labels'] == sess_label

        if not np.any(day_mask):
            continue

        for t, tname in enumerate(type_names):
            # ── Cell-level chi-squared ──────────────────────────
            contingency = []
            for layer in layer_list:
                lm  = day_mask & (pooled['layer_labels'] == layer)
                n   = int(np.sum(lm))
                if n == 0:
                    continue
                n_in  = int(np.sum(raw_labels[lm] == t))
                n_out = n - n_in
                contingency.append([n_in, n_out])

            p_cell, chi2_val, n_cells = np.nan, np.nan, int(np.sum(day_mask))
            if len(contingency) >= 2:
                try:
                    table = np.array(contingency)
                    chi2_val, p_cell, _, _ = chi2_contingency(table)
                except Exception:
                    pass

            # ── Per-animal KW + Fisher's ────────────────────────
            per_animal_p = []
            for animal in pooled['animal_ids']:
                amask = pooled['animal_labels'] == animal
                mask  = amask & (pooled['session_labels'] == sess_label)
                if not np.any(mask):
                    continue
                al_d = pooled['layer_labels'][mask]
                rl_d = raw_labels[mask]
                groups = []
                for layer in layer_list:
                    lm = al_d == layer
                    if np.sum(lm) == 0:
                        continue
                    groups.append((rl_d[lm] == t).astype(float))
                if len(groups) < 2:
                    continue
                try:
                    _, kw_p = kruskal(*groups)
                    per_animal_p.append(kw_p)
                except Exception:
                    pass

            p_fisher  = np.nan
            if len(per_animal_p) >= 2:
                _, p_fisher = combine_pvalues(per_animal_p, method='fisher')
            elif len(per_animal_p) == 1:
                p_fisher = per_animal_p[0]

            results[tname][day] = {
                'p_cell':    p_cell,
                'chi2':      chi2_val,
                'n_cells':   n_cells,
                'p_fisher':  p_fisher,
                'n_animals': len(per_animal_p),
            }

    # Print summary
    for tname in type_names:
        print(f"\n  {tname}:")
        for day in all_days:
            entry = results[tname].get(day)
            if entry is None:
                continue
            pc = entry['p_cell']
            pf = entry['p_fisher']
            print(f"    Day{day:>2d} | cell χ² p={pc:.4f} {_sig_stars(pc)}"
                  f"  | Fisher p={pf:.4f} {_sig_stars(pf)}"
                  f"  | n={entry['n_cells']} cells, {entry['n_animals']} animals")

    return results, all_days


def plot_layer_posthoc_per_day(pooled, raw_labels, per_session_results,
                                all_days, layer_list, n_types,
                                type_names, type_colors, output_dir=None):
    """
    Same format as plot_layer_posthoc_pooled but one figure per recording day.
    Each figure: one panel per cell type, bars = % cells per layer,
    per-animal dots, Bonferroni-corrected pairwise chi-squared brackets,
    omnibus chi-squared p (cell-level) in x-axis label.
    Saved as layer_posthoc_Day{N}.png if output_dir is given.
    """
    figs = {}
    for day in sorted(all_days):
        sess_label = f'Day{day}'
        day_mask   = pooled['session_labels'] == sess_label
        if not np.any(day_mask):
            continue

        fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 5))
        fig.suptitle(f'Layer effect — {sess_label} (pooled cells + Bonferroni)',
                     fontweight='bold')
        if n_types == 1:
            axes = [axes]

        xpos = np.arange(len(layer_list))

        for t, (tname, tcolor, ax) in enumerate(zip(type_names, type_colors, axes)):
            # Grand-mean bars (all pooled cells this day)
            grand_mean, ns = [], []
            for layer in layer_list:
                lm = day_mask & (pooled['layer_labels'] == layer)
                n  = int(np.sum(lm))
                ns.append(n)
                grand_mean.append(np.mean(raw_labels[lm] == t) * 100 if n > 0 else 0)
            ax.bar(xpos, grand_mean, color=tcolor, alpha=0.65, width=0.6, zorder=2)
            for xi, n in enumerate(ns):
                ax.text(xi, grand_mean[xi] + 1, f'n={n}', ha='center',
                        va='bottom', fontsize=7, rotation=45)

            # Per-animal dots
            for animal in pooled['animal_ids']:
                amask = pooled['animal_labels'] == animal
                mask  = amask & day_mask
                if not np.any(mask):
                    continue
                al = pooled['layer_labels'][mask]
                rl = raw_labels[mask]
                vals = []
                for layer in layer_list:
                    lm = al == layer
                    n  = int(np.sum(lm))
                    vals.append(np.mean(rl[lm] == t) * 100 if n > 0 else np.nan)
                ax.scatter(xpos, vals, color='black',
                           marker=ANIMAL_MARKERS.get(animal, 'o'),
                           s=50, zorder=3, clip_on=False)

            # Pairwise chi-squared + Bonferroni on pooled cells this day
            groups, grp_layers = [], []
            for layer in layer_list:
                lm = day_mask & (pooled['layer_labels'] == layer)
                if np.sum(lm) == 0:
                    continue
                groups.append((raw_labels[lm] == t).astype(float))
                grp_layers.append(layer)

            pairs   = [(i, j) for i in range(len(groups))
                               for j in range(i + 1, len(groups))]
            n_pairs = len(pairs)
            sig_pairs = []
            for i, j in pairs:
                try:
                    n_in_i  = int(np.sum(groups[i]))
                    n_out_i = len(groups[i]) - n_in_i
                    n_in_j  = int(np.sum(groups[j]))
                    n_out_j = len(groups[j]) - n_in_j
                    table   = np.array([[n_in_i, n_out_i], [n_in_j, n_out_j]])
                    _, p_raw, _, _ = chi2_contingency(table)
                    p_adj = min(p_raw * n_pairs, 1.0)
                    if p_adj < 0.05:
                        xi = layer_list.index(grp_layers[i])
                        xj = layer_list.index(grp_layers[j])
                        sig_pairs.append((xi, xj, _sig_stars(p_adj)))
                except Exception:
                    pass

            y_top = max(grand_mean) if grand_mean else 0
            step  = max(y_top * 0.13, 2.0)
            base  = y_top * 1.08
            placed = []
            for xi, xj, stars in sorted(sig_pairs, key=lambda x: abs(x[1] - x[0])):
                level = 0
                while any(not (xj < px or xi > pxj) and lv == level
                          for px, pxj, lv in placed):
                    level += 1
                placed.append((xi, xj, level))
                _draw_sig_bracket(ax, xi, xj, base + level * step, step * 0.3, stars)

            ax.set_xticks(xpos)
            ax.set_xticklabels(layer_list, fontsize=9)
            ax.set_title(tname, color=tcolor, fontweight='bold')
            ax.set_ylabel('% cells' if t == 0 else '')
            ax.set_ylim(0, None)
            ax.grid(True, alpha=0.2, axis='y')

            entry   = per_session_results.get(tname, {}).get(day, {})
            p_omni  = entry.get('p_cell', np.nan)
            p_fish  = entry.get('p_fisher', np.nan)
            ax.set_xlabel(
                f'χ² p={p_omni:.3f} {_sig_stars(p_omni)}  |  '
                f"Fisher p={p_fish:.3f} {_sig_stars(p_fish)}",
                fontsize=7)

        plt.tight_layout()
        figs[day] = fig
        if output_dir:
            path = os.path.join(output_dir, f'layer_posthoc_{sess_label}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')

    return figs


def plot_layer_posthoc_grand_summary(pooled, raw_labels, per_session_results,
                                      all_days, layer_list, n_types,
                                      type_names, type_colors, output_path=None):
    """
    Grand summary: one panel per cell type.
    x = recording day, one line per layer showing % of cells in that cluster.
    Significant days (cell-level χ² p < 0.05) marked with a star on the x-axis.
    Shows whether layer differences in cluster membership change with experience.
    """
    n_types  = len(type_names)
    days_arr = np.array(sorted(all_days), dtype=int)
    day_labels = [f'Day{d}' for d in days_arr]

    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4), sharey=False)
    fig.suptitle('Layer × cell type proportions across sessions (all pooled cells)',
                 fontweight='bold')
    if n_types == 1:
        axes = [axes]

    for t, (tname, tcolor, ax) in enumerate(zip(type_names, type_colors, axes)):
        for layer in layer_list:
            lcolor = LAYER_COLORS.get(layer, 'gray')
            props  = []
            for day in days_arr:
                sess_label = f'Day{day}'
                mask = (pooled['session_labels'] == sess_label) & \
                       (pooled['layer_labels'] == layer)
                n    = int(np.sum(mask))
                props.append(np.mean(raw_labels[mask] == t) * 100 if n > 0 else np.nan)
            ax.plot(range(len(days_arr)), props, 'o-', color=lcolor,
                    linewidth=2, markersize=5, label=layer)

        # # Mark days where layer effect is significant (cell-level)
        # for di, day in enumerate(days_arr):
        #     p = per_session_results.get(tname, {}).get(day, {}).get('p_cell', np.nan)
        #     if not np.isnan(p) and p < 0.05:
        #         ax.text(di, ax.get_ylim()[0], _sig_stars(p),
        #                 ha='center', va='bottom', fontsize=9,
        #                 color='black', fontweight='bold')

        ax.set_xticks(range(len(days_arr)))
        ax.set_xticklabels(day_labels, rotation=45, ha='right', fontsize=8)
        ax.set_title(tname, color=tcolor, fontweight='bold')
        ax.set_ylabel('% cells' if t == 0 else '')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, title='Layer')
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_layer_effect_cell_level(pooled, raw_labels, layer_cell_results,
                                  layer_list, n_types, type_names, type_colors,
                                  output_path=None):
    """
    One panel per cell type: bar = % of cells in that cluster per layer,
    with pairwise chi-squared post-hoc significance brackets (Bonferroni).
    Omnibus chi-squared p shown in x-axis label.
    Unit of analysis = individual cells (not animals).
    """
    xpos = np.arange(len(layer_list))
    fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 5))
    fig.suptitle('Layer effect on cell type — cell-level chi-squared (all pooled cells)',
                 fontweight='bold')
    if n_types == 1:
        axes = [axes]

    for t, (tname, tcolor, ax) in enumerate(zip(type_names, type_colors, axes)):
        # Proportion bar per layer
        pcts, ns = [], []
        for layer in layer_list:
            lm = pooled['layer_labels'] == layer
            n  = int(np.sum(lm))
            ns.append(n)
            pcts.append(np.mean(raw_labels[lm] == t) * 100 if n > 0 else 0)

        ax.bar(xpos, pcts, color=tcolor, alpha=0.75, width=0.6, zorder=2)
        for xi, (pct, n) in enumerate(zip(pcts, ns)):
            ax.text(xi, pct + 1, f'n={n}', ha='center', va='bottom',
                    fontsize=7, rotation=45)

        # Sig brackets from pairwise post-hoc
        entry    = layer_cell_results.get(tname, {})
        posthoc  = entry.get('posthoc', {})
        p_omni   = entry.get('p_omni', np.nan)
        sig_pairs = []
        for (la, lb), p_adj in posthoc.items():
            if not np.isnan(p_adj) and p_adj < 0.05:
                xi = layer_list.index(la)
                xj = layer_list.index(lb)
                sig_pairs.append((xi, xj, _sig_stars(p_adj)))

        y_top = max(pcts) if pcts else 0
        step  = max(y_top * 0.13, 2.0)
        base  = y_top * 1.08
        placed = []
        for xi, xj, stars in sorted(sig_pairs, key=lambda x: abs(x[1] - x[0])):
            level = 0
            while any(not (xj < px or xi > pxj) and lv == level
                      for px, pxj, lv in placed):
                level += 1
            placed.append((xi, xj, level))
            _draw_sig_bracket(ax, xi, xj, base + level * step, step * 0.3, stars)

        ax.set_xticks(xpos)
        ax.set_xticklabels(layer_list, fontsize=9)
        ax.set_title(tname, color=tcolor, fontweight='bold')
        ax.set_ylabel('% cells' if t == 0 else '')
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.2, axis='y')
        omni_str = (f'χ² p={p_omni:.4e} {_sig_stars(p_omni)}'
                    if not np.isnan(p_omni) else '')
        ax.set_xlabel(omni_str, fontsize=8)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_experience_pooled_regression(exp_pooled_results, type_names, type_colors,
                                       layer_list, output_path=None):
    """
    Grid: rows = cell types, columns = layers.
    Each panel: scatter of all pooled (animal × session) data points + OLS line.
    """
    n_types  = len(type_names)
    n_layers = len(layer_list)

    fig, axes = plt.subplots(n_types, n_layers,
                             figsize=(4 * n_layers, 3.5 * n_types),
                             sharey='row', sharex='col')
    fig.suptitle('Experience effect — pooled cell-level OLS (all animals)',
                 fontweight='bold')

    if n_types == 1:
        axes = axes[np.newaxis, :]
    if n_layers == 1:
        axes = axes[:, np.newaxis]

    for t, (tname, tcolor) in enumerate(zip(type_names, type_colors)):
        for li, layer in enumerate(layer_list):
            ax    = axes[t, li]
            entry = exp_pooled_results[tname].get(layer, {})
            days  = entry.get('days')
            props = entry.get('props')
            slope     = entry.get('slope', np.nan)
            intercept = entry.get('intercept', np.nan)
            p         = entry.get('p', np.nan)
            r2        = entry.get('r2', np.nan)

            lcolor = LAYER_COLORS.get(layer, 'gray')

            if days is not None and len(days) >= 3:
                ax.scatter(days, props, color=tcolor, s=25, alpha=0.7, zorder=3)
                x_line = np.linspace(days.min(), days.max(), 50)
                ax.plot(x_line, slope * x_line + intercept,
                        color=lcolor,
                        linewidth=2 if p < 0.05 else 1,
                        linestyle='-' if p < 0.05 else '--')
                ax.set_title(
                    f'{tname[:10]} | {layer}\n'
                    f'slope={slope:+.2f}  r²={r2:.2f}  {_sig_stars(p)}',
                    fontsize=7, color=tcolor if p < 0.05 else 'black')
            else:
                ax.set_title(f'{tname[:10]} | {layer}\n(insufficient data)',
                             fontsize=7)

            if li == 0:
                ax.set_ylabel('% cells', fontsize=8)
            if t == n_types - 1:
                ax.set_xlabel('Recording day', fontsize=8)
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


# ── Layer × session interaction on pooled cells ───────────────────────────────

def run_layer_session_interaction(pooled, raw_labels, layer_list, type_names):
    """
    For each cluster × layer: KW test across all sessions (pooled cells),
    then Bonferroni-corrected pairwise Mann-Whitney U between every pair of days.

    Returns
    -------
    results : dict  {tname: {layer: {
        'kw_p':     float,
        'kw_H':     float,
        'n_days':   int,
        'posthoc':  dict { (day_a, day_b): {'p_raw', 'p_bonf', 'n_a', 'n_b'} }
    }}}
    """
    from scipy.stats import kruskal, mannwhitneyu

    # Collect all unique day numbers
    all_days = sorted({int(s.replace('Day', ''))
                       for a in pooled['animal_ids']
                       for s in pooled['animal_data'][a]['session_order']})

    print("\n── Layer × session interaction (KW + Bonferroni MW-U, pooled cells) ──")
    results = {}

    for t, tname in enumerate(type_names):
        results[tname] = {}
        print(f"\n  {tname}:")

        for layer in layer_list:
            lmask = pooled['layer_labels'] == layer

            # Build per-day binary membership arrays (1 = in cluster t)
            day_groups = {}
            for day in all_days:
                dmask = pooled['session_labels'] == f'Day{day}'
                mask  = lmask & dmask
                if np.sum(mask) < 2:
                    continue
                day_groups[day] = (raw_labels[mask] == t).astype(float)

            if len(day_groups) < 2:
                results[tname][layer] = {'kw_p': np.nan, 'kw_H': np.nan,
                                         'n_days': 0, 'posthoc': {}}
                continue

            days_present = sorted(day_groups)
            groups       = [day_groups[d] for d in days_present]

            # KW omnibus
            try:
                H, kw_p = kruskal(*groups)
            except Exception:
                H, kw_p = np.nan, np.nan

            # Pairwise MW-U with Bonferroni correction
            pairs = [(days_present[i], days_present[j])
                     for i in range(len(days_present))
                     for j in range(i + 1, len(days_present))]
            n_pairs  = len(pairs)
            posthoc  = {}
            for da, db in pairs:
                ga, gb = day_groups[da], day_groups[db]
                try:
                    _, p_raw = mannwhitneyu(ga, gb, alternative='two-sided')
                except Exception:
                    p_raw = np.nan
                posthoc[(da, db)] = {
                    'p_raw':  p_raw,
                    'p_bonf': min(p_raw * n_pairs, 1.0) if not np.isnan(p_raw) else np.nan,
                    'n_a':    len(ga),
                    'n_b':    len(gb),
                }

            results[tname][layer] = {
                'kw_p':    kw_p,
                'kw_H':    H,
                'n_days':  len(days_present),
                'posthoc': posthoc,
            }
            sig = _sig_stars(kw_p)
            print(f"    {layer}: KW H={H:.2f}  p={kw_p:.4f}  {sig}  "
                  f"(n_days={len(days_present)})")

    return results, all_days


def plot_layer_session_interaction(results, pooled, raw_labels, all_days,
                                   layer_list, type_names, type_colors,
                                   output_path=None):
    """
    Grid: rows = clusters, columns = layers.
    Each panel: mean proportion per day (line) with significant pairwise
    day comparisons annotated as brackets above.
    Background tinted by KW significance.
    """
    n_types  = len(type_names)
    n_layers = len(layer_list)
    x        = np.array(all_days)

    fig, axes = plt.subplots(n_layers, n_types,
                             figsize=(3.5* n_types, 4.5 * n_layers),
                             sharey='col', sharex='col')
    fig.suptitle('Cluster proportion across sessions per layer',
                 fontweight='bold', y=1.01)

    if n_layers == 1:
        axes = axes[np.newaxis, :]
    if n_types == 1:
        axes = axes[:, np.newaxis]

    for ti, (tname, tcolor) in enumerate(zip(type_names, type_colors)):
        for li, layer in enumerate(layer_list):
            ax   = axes[li, ti]
            res  = results[tname].get(layer, {})
            kw_p = res.get('kw_p', np.nan)

            # # Tint background if KW significant
            # if not np.isnan(kw_p) and kw_p < 0.05:
            #     ax.set_facecolor('#fff8e1' if kw_p < 0.01 else '#fffde7')

            # Per-animal proportion per day — one point per animal per day
            lmask      = pooled['layer_labels'] == layer
            animal_ids = pooled['animal_ids']
            day_means  = []

            for day in all_days:
                dmask = pooled['session_labels'] == f'Day{day}'
                animal_props = []
                for animal in animal_ids:
                    amask = pooled['animal_labels'] == animal
                    mask  = lmask & dmask & amask
                    n     = int(np.sum(mask))
                    if n == 0:
                        continue
                    animal_props.append(np.mean((raw_labels[mask] == ti).astype(float)) * 100)
                    ax.scatter(day, animal_props[-1], color=tcolor, s=18,
                               alpha=0.6, zorder=3, linewidths=0)
                day_means.append(np.nanmean(animal_props) if animal_props else np.nan)

            m     = np.array(day_means)
            valid = ~np.isnan(m)
            ax.plot(x[valid], m[valid], '-', color=tcolor, linewidth=2, zorder=2)

            if li == 0:
                ax.set_title(tname, fontsize=15, color=tcolor, fontweight='bold')
            if ti == 0:
                ax.set_ylabel(f'{layer}\n% cells', fontsize=15)
            if li == n_layers - 1:
                ax.set_xlabel('Day', fontsize=15)
            ax.set_xticks(x)
            ax.tick_params(axis='both', labelsize=15)
            ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


# ── Cluster proportion by day (pooled cells, mean±std across animals) ─────────

def compute_cluster_proportion_by_day(pooled, raw_labels, type_names):
    """
    For each cluster and each recording day, compute the proportion of cells
    belonging to that cluster — separately per animal, then summarise as
    mean ± std across animals.

    Returns
    -------
    results : dict  {tname: {'days': array, 'mean': array, 'std': array,
                              'per_animal': dict {animal: array}}}
    all_days : sorted list of int day numbers
    """
    # Collect all unique day numbers across animals
    day_set = set()
    for animal in pooled['animal_ids']:
        so = pooled['animal_data'][animal]['session_order']
        for sess in so:
            day_set.add(int(sess.replace('Day', '')))
    all_days = sorted(day_set)

    results = {}
    for t, tname in enumerate(type_names):
        per_animal = {}
        for animal in pooled['animal_ids']:
            amask = pooled['animal_labels'] == animal
            sl    = pooled['session_labels'][amask]
            rl    = raw_labels[amask]
            so    = pooled['animal_data'][animal]['session_order']

            animal_props = {}
            for sess in so:
                day  = int(sess.replace('Day', ''))
                mask = sl == sess
                n    = int(np.sum(mask))
                if n == 0:
                    continue
                animal_props[day] = np.mean(rl[mask] == t) * 100
            per_animal[animal] = animal_props

        # Align across animals for each day
        mean_arr = np.full(len(all_days), np.nan)
        std_arr  = np.full(len(all_days), np.nan)
        for di, day in enumerate(all_days):
            vals = [per_animal[a][day] for a in pooled['animal_ids']
                    if day in per_animal[a]]
            if len(vals) >= 2:
                mean_arr[di] = np.mean(vals)
                std_arr[di]  = np.std(vals, ddof=1)
            elif len(vals) == 1:
                mean_arr[di] = vals[0]
                std_arr[di]  = 0.0

        results[tname] = {
            'days':       np.array(all_days),
            'mean':       mean_arr,
            'std':        std_arr,
            'per_animal': per_animal,
        }

    return results, all_days


def plot_cluster_proportion_by_day(prop_results, all_days, type_names, type_colors,
                                   output_path=None):
    """
    Two panels:
      Left  — one line per cluster, mean ± std (shaded) across animals over days
      Right — same data normalised so all clusters sum to 100% per day (stacked area)
    """
    x = np.array(all_days)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Cluster proportion across experience (pooled cells, mean ± SD across animals)',
                 fontweight='bold')

    # ── Left: mean ± std lines ────────────────────────────────
    ax = axes[0]
    for tname, tcolor in zip(type_names, type_colors):
        d = prop_results[tname]
        m = d['mean']
        s = d['std']
        valid = ~np.isnan(m)
        ax.plot(x[valid], m[valid], 'o-', color=tcolor, label=tname, linewidth=2, markersize=5)
        ax.fill_between(x[valid], (m - s)[valid], (m + s)[valid],
                        color=tcolor, alpha=0.15)

    ax.set_xlabel('Recording day')
    ax.set_ylabel('% cells in cluster')
    ax.set_title('Mean ± SD across animals')
    ax.set_xticks(x)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2, axis='y')

    # ── Right: stacked area (normalised) ─────────────────────
    ax2 = axes[1]
    # Build matrix (n_types, n_days), fill NaN days with 0 for stacking
    mat = np.vstack([prop_results[tn]['mean'] for tn in type_names])
    mat_norm = np.where(np.isnan(mat), 0, mat)
    totals   = mat_norm.sum(axis=0, keepdims=True)
    totals   = np.where(totals == 0, 1, totals)
    mat_pct  = mat_norm / totals * 100

    bottom = np.zeros(len(x))
    for ti, (tname, tcolor) in enumerate(zip(type_names, type_colors)):
        ax2.fill_between(x, bottom, bottom + mat_pct[ti],
                         color=tcolor, alpha=0.75, label=tname, step='mid')
        bottom += mat_pct[ti]

    ax2.set_xlabel('Recording day')
    ax2.set_ylabel('% cells (normalised)')
    ax2.set_title('Relative cluster composition per day')
    ax2.set_xticks(x)
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=8, framealpha=0.9, loc='upper right')
    ax2.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


# ── Save cluster labels back to each animal's HDF5 ────────────

def save_labels_to_h5(pooled, raw_labels, type_names, pca, kmeans):
    """
    Write cluster assignments + PCA/k-means objects back to each animal's
    *_pca_data.h5 so that downstream scripts (CellType_Evolution.py etc.)
    can reload the model without re-fitting.
    """
    for animal, filepath in ANIMALS.items():
        if animal not in pooled['animal_ids']:
            continue
        amask  = pooled['animal_labels'] == animal
        a_labels = raw_labels[amask]
        a_profiles = pooled['profiles'][amask]

        print(f"  Writing {animal} → {filepath}")
        with h5py.File(filepath, 'a') as f:
            for grp in ('pca_results', 'clustering'):
                if grp in f:
                    del f[grp]

            pr = f.create_group('pca_results')
            pr.create_dataset('components',               data=pca.components_)
            pr.create_dataset('explained_variance_ratio', data=pca.explained_variance_ratio_)
            pr.create_dataset('mean_',                    data=pca.mean_)
            pr.create_dataset('pc_scores',                data=pca.transform(a_profiles))

            cl = f.create_group('clustering')
            cl.create_dataset('raw_labels',     data=a_labels)
            cl.create_dataset('kmeans_centers', data=kmeans.cluster_centers_)
            cl.attrs['n_types'] = kmeans.n_clusters
            cl.create_dataset('type_names',
                              data=np.array(type_names, dtype='S40'))
        print(f"    Done  ({int(np.sum(amask))} cells, "
              f"{kmeans.n_clusters} types)")


# ── Main ──────────────────────────────────────────────────────

if __name__ == '__main__':

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load + pool ────────────────────────────────────────
    print("=== 1. Load and pool data ===")
    pooled = load_all_animals(ANIMALS)

    profiles   = pooled['profiles']
    bin_centers        = pooled['bin_centers']
    landmark_positions = pooled['landmark_positions']
    layer_list = [l for l in LAYER_ORDER
                  if np.any(pooled['layer_labels'] == l)]

    # ── 2. PCA ────────────────────────────────────────────────
    print("\n=== 2. PCA ===")
    pca, pc_scores = run_pca(profiles, N_PCA_COMPONENTS)

    # ── 3. K selection + clustering ───────────────────────────
    print("\n=== 3. K selection + clustering ===")
    X = pc_scores[:, :N_CLUSTER_PCS]
    sil_scores, inertias, auto_k = select_optimal_k(X, K_RANGE)
    optimal_k = OVERRIDE_K if OVERRIDE_K is not None else auto_k
    print(f"  K: auto={auto_k}  used={optimal_k}")

    kmeans, raw_labels = fit_kmeans(X, optimal_k)
    n_types = optimal_k

    # ── 4. Semantic labels ────────────────────────────────────
    print("\n=== 4. Semantic labels ===")
    type_names, type_colors, mean_profiles = assign_semantic_labels(
        raw_labels, profiles, n_types, bin_centers, landmark_positions
    )
    for t in range(n_types):
        n = int(np.sum(raw_labels == t))
        print(f"  [{t}] {type_names[t]:25s}  n={n} ({n/len(raw_labels)*100:.1f}%)")

    # ── 5. Population figures ─────────────────────────────────
    print("\n=== 5. Population figures ===")
    plot_scree(pca, N_CLUSTER_PCS,
               os.path.join(OUTPUT_DIR, 'pooled_scree.png'))

    plot_k_selection(sil_scores, inertias, K_RANGE, optimal_k,
                     os.path.join(OUTPUT_DIR, 'pooled_k_selection.png'))

    plot_pc_scatter(pc_scores, raw_labels, pooled['animal_labels'],
                    type_names, type_colors, pooled['animal_ids'],
                    os.path.join(OUTPUT_DIR, 'pooled_pc_scatter.png'))

    plot_mean_profiles(mean_profiles, bin_centers, landmark_positions,
                       type_names, type_colors,
                       profiles=profiles, raw_labels=raw_labels,
                       output_path=os.path.join(OUTPUT_DIR,
                                                'pooled_mean_profiles.png'))

    # # ── 5b. Adaptation-like diagnostic ───────────────────────
    # print("\n=== 5b. Adaptation-like diagnostic ===")
    # plot_adaptation_cells_diagnostic(
    #     profiles, bin_centers, landmark_positions,
    #     raw_labels, type_names,
    #     label='pooled', n_cells=50, cells_per_fig=10,
    #     output_dir=OUTPUT_DIR
    # )

    # ── 6. Per-animal proportion figures ──────────────────────
    print("\n=== 6. Per-animal figures ===")
    plot_per_animal_stacked_bars(
        pooled, raw_labels, layer_list, type_names, type_colors,
        output_path=os.path.join(OUTPUT_DIR, 'per_animal_stacked_bars.png')
    )

    plot_per_animal_trajectories(
        pooled, raw_labels, layer_list, type_names, type_colors,
        output_path_template=os.path.join(OUTPUT_DIR,
                                          '{animal}_trajectories.png')
    )

    # ── 7. Cross-animal summary figure ────────────────────────
    print("\n=== 7. Cross-animal summary ===")
    plot_pooled_proportions_by_layer(
        pooled, raw_labels, layer_list, n_types, type_names, type_colors,
        output_path=os.path.join(OUTPUT_DIR, 'pooled_proportions_by_layer.png')
    )

    # ── 8. Statistical tests ──────────────────────────────────
    print("\n=== 8. Statistical tests ===")
    # Animal-level: Fisher's method combining per-animal tests
    layer_results = run_layer_effect_fisher(
        pooled, raw_labels, layer_list, n_types, type_names)
    exp_results   = run_experience_effect_fisher(
        pooled, raw_labels, layer_list, type_names)
    trend_results = run_session_trend_tests_fisher(
        pooled, raw_labels, layer_list, type_names)

    # Cell-level: tests on all pooled cells / pooled sessions
    layer_cell_results = run_layer_effect_pooled_cells(
        pooled, raw_labels, layer_list, n_types, type_names)
    exp_cell_results   = run_experience_effect_pooled_cells(
        pooled, raw_labels, layer_list, type_names)

    # ── 9. Statistical figures ────────────────────────────────
    print("\n=== 9. Statistical figures ===")
    plot_layer_posthoc_pooled(
        pooled, raw_labels, layer_list, n_types, type_names, type_colors,
        output_path=os.path.join(OUTPUT_DIR, 'pooled_layer_posthoc.png')
    )

    plot_layer_effect_summary(
        layer_results, type_names, pooled['animal_ids'],
        output_path=os.path.join(OUTPUT_DIR, 'layer_effect_summary.png')
    )

    plot_experience_regression_all_animals(
        pooled, raw_labels, layer_list, type_names, type_colors,
        output_path=os.path.join(OUTPUT_DIR, 'experience_regression_all_animals.png')
    )

    plot_experience_summary_heatmap(
        exp_results, type_names, layer_list, pooled['animal_ids'],
        output_path=os.path.join(OUTPUT_DIR, 'experience_summary_heatmap.png')
    )

    plot_layer_effect_cell_level(
        pooled, raw_labels, layer_cell_results, layer_list,
        n_types, type_names, type_colors,
        output_path=os.path.join(OUTPUT_DIR, 'layer_effect_cell_level.png')
    )

    per_session_results, all_days = run_layer_effect_per_session(
        pooled, raw_labels, layer_list, n_types, type_names)
    plot_layer_posthoc_per_day(
        pooled, raw_labels, per_session_results, all_days,
        layer_list, n_types, type_names, type_colors,
        output_dir=OUTPUT_DIR
    )
    plot_layer_posthoc_grand_summary(
        pooled, raw_labels, per_session_results, all_days,
        layer_list, n_types, type_names, type_colors,
        output_path=os.path.join(OUTPUT_DIR, 'layer_effect_grand_summary.png')
    )

    plot_experience_pooled_regression(
        exp_cell_results, type_names, type_colors, layer_list,
        output_path=os.path.join(OUTPUT_DIR, 'experience_pooled_regression.png')
    )

    plot_session_trend_across_animals(
        trend_results, pooled, layer_list, type_names, type_colors,
        output_path=os.path.join(OUTPUT_DIR, 'session_trend_across_animals.png')
    )

    prop_by_day, all_days_pooled = compute_cluster_proportion_by_day(
        pooled, raw_labels, type_names)
    plot_cluster_proportion_by_day(
        prop_by_day, all_days_pooled, type_names, type_colors,
        output_path=os.path.join(OUTPUT_DIR, 'cluster_proportion_by_day.png')
    )

    layer_sess_results, all_days_int = run_layer_session_interaction(
        pooled, raw_labels, layer_list, type_names)
    plot_layer_session_interaction(
        layer_sess_results, pooled, raw_labels, all_days_int,
        layer_list, type_names, type_colors,
        output_path=os.path.join(OUTPUT_DIR, 'layer_session_interaction.png')
    )

    # ── 10. Save labels to each animal's HDF5 ─────────────────
    print("\n=== 10. Save labels to HDF5 ===")
    save_labels_to_h5(pooled, raw_labels, type_names, pca, kmeans)

    print(f"\n=== Done — figures saved to: {OUTPUT_DIR} ===")
    plt.show()
