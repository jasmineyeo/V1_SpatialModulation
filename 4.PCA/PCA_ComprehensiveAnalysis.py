"""
PCA_ComprehensiveAnalysis.py
============================
Population-level PCA + k-means functional cell type analysis.

Loads aggregated spatial profiles produced by PCA_DataAggregation.py, fits a
PCA + k-means model on ALL reliable cells (all layers, all landmarks), assigns
semantic labels to clusters, and generates a full set of analysis figures.

Questions addressed:
  1. How many functional cell types are there and what do their spatial profiles
     look like?
  2. Do cell type proportions differ across cortical layers?
  3. Do proportions change with experience (across recording days)?

Prerequisite: run PCA_DataAggregation.py first to produce *_pca_data.h5.
The cluster labels + PCA/KMeans objects are saved back to that HDF5 for
downstream use by 6.TrackedROIAnalysis/CellType_Evolution.py.

JSY, 2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import chi2_contingency, kruskal
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# CONFIGURATION
# ============================================================

PCA_DATA_FILE    = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\PCA\JSY054_pca_data.h5"

N_PCA_COMPONENTS = 10      # total PCs to fit
N_CLUSTER_PCS    = 5       # top PCs used for k-means
K_RANGE          = range(2, 8)
OVERRIDE_K       = None    # set to int to force a specific k

OUTPUT_DIR = None          # None → same folder as this script

# ============================================================

LAYER_ORDER  = ['L2/3', 'L4', 'L5', 'L6']
LAYER_COLORS = {'L2/3': '#4CAF50', 'L4': '#2196F3', 'L5': '#FF9800', 'L6': '#9C27B0'}


# ── Data loading ──────────────────────────────────────────────

def load_pca_data(filepath):
    """
    Load aggregated data from *_pca_data.h5.

    Returns
    -------
    profiles    : (n_cells, n_bins) z-scored spatial profiles
    bin_centers : (n_bins,) in cm
    landmark_positions : (n_landmarks,) in cm
    session_labels     : (n_cells,) str e.g. 'Day1'
    layer_labels       : (n_cells,) str e.g. 'L2/3'
    session_order      : list of session IDs sorted by day number
    animal_id          : str
    """
    with h5py.File(filepath, 'r') as f:
        # Prefer session-corrected profiles if present; otherwise use z-scored
        if 'features/spatial_profiles_session_corrected' in f:
            profiles = f['features/spatial_profiles_session_corrected'][:]
        else:
            profiles = f['features/spatial_profiles_zscore'][:]

        bin_centers        = f['metadata/bin_centers_trimmed'][:]
        landmark_positions = f['metadata/landmark_positions'][:]
        animal_id          = f['metadata'].attrs['animal_id']

        session_labels = f['cells/session_labels'][:].astype(str)
        layer_labels   = f['cells/layer_labels'][:].astype(str)

        raw_session_ids = f['metadata/session_ids'][:].astype(str)

    animal_id     = animal_id if isinstance(animal_id, str) else animal_id.decode()
    session_order = sorted(raw_session_ids.tolist(),
                           key=lambda s: int(s.replace('Day', '')))

    print(f"  Animal: {animal_id}  |  {len(profiles)} cells  |  "
          f"{len(session_order)} sessions")
    return profiles, bin_centers, landmark_positions, session_labels, \
           layer_labels, session_order, animal_id


# ── PCA + clustering ──────────────────────────────────────────

def run_pca(profiles, n_components):
    """Fit PCA; return (pca, pc_scores)."""
    pca    = PCA(n_components=n_components)
    scores = pca.fit_transform(profiles)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    print(f"  Variance explained: "
          + "  ".join([f"PC{i+1}={pca.explained_variance_ratio_[i]*100:.1f}%"
                       for i in range(min(5, n_components))])
          + f"  |  top-{n_components} cumulative: {cumvar[-1]:.1f}%")
    return pca, scores


def select_optimal_k(X, k_range, random_state=42):
    """
    Silhouette analysis to pick the best k.
    Returns (sil_scores, inertias, auto_k).
    """
    sil_scores, inertias = [], []
    for k in k_range:
        km  = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        lbs = km.fit_predict(X)
        sil_scores.append(silhouette_score(X, lbs))
        inertias.append(km.inertia_)
    auto_k = list(k_range)[int(np.argmax(sil_scores))]
    return sil_scores, inertias, auto_k


def fit_kmeans(X, k, random_state=42):
    """Fit k-means on feature matrix X; return (kmeans, raw_labels)."""
    km     = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(X)
    return km, labels


def assign_semantic_labels(raw_labels, profiles, n_types,
                            bin_centers, landmark_positions):
    """
    Map cluster IDs to human-readable names based on mean profile shape.
      Earliest peak  → Adaptation-like
      Latest peak    → L4-preferring
      Most uniform   → Visually responsive
      Remaining      → Peak~{cm}cm
    Returns (type_names, type_colors, mean_profiles).
    """
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


# ── Proportion matrix ─────────────────────────────────────────

def build_prop_matrix(raw_labels, session_labels, layer_labels,
                       session_order, layer_list, n_types):
    """
    Build (n_types, n_layers, n_sessions) proportion matrix.
    Also returns (n_layers, n_sessions) count matrix.
    """
    n_sessions = len(session_order)
    n_layers   = len(layer_list)
    prop   = np.full((n_types, n_layers, n_sessions), np.nan)
    counts = np.zeros((n_layers, n_sessions), dtype=int)

    for si, sess in enumerate(session_order):
        sess_mask = session_labels == sess
        for li, layer in enumerate(layer_list):
            mask  = sess_mask & (layer_labels == layer)
            n     = int(np.sum(mask))
            if n == 0:
                continue
            counts[li, si] = n
            for t in range(n_types):
                prop[t, li, si] = np.sum(raw_labels[mask] == t) / n

    return prop, counts


# ── Plotting ──────────────────────────────────────────────────

def plot_scree(pca, n_cluster_pcs, animal_id, output_path=None):
    """Variance explained scree plot."""
    var = pca.explained_variance_ratio_ * 100
    cumvar = np.cumsum(var)
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle(f'{animal_id} — PCA Scree', fontweight='bold')
    ax.bar(range(1, len(var) + 1), var, color='steelblue', alpha=0.8)
    ax.plot(range(1, len(var) + 1), cumvar, 'ko-', markersize=4)
    ax.axvline(n_cluster_pcs + 0.5, color='red', linestyle='--',
               label=f'Clustering uses top {n_cluster_pcs} PCs')
    ax.set_xlabel('PC')
    ax.set_ylabel('Variance explained (%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_k_selection(sil_scores, inertias, k_range, optimal_k,
                     animal_id, output_path=None):
    """K selection: silhouette + elbow."""
    ks = list(k_range)
    fig, ax1 = plt.subplots(figsize=(5, 4))
    fig.suptitle(f'{animal_id} — K selection', fontweight='bold')
    ax2 = ax1.twinx()
    ax1.plot(ks, sil_scores, 'b-o', label='Silhouette', linewidth=2)
    ax2.plot(ks, inertias,   'r-s', label='Inertia',    linewidth=2)
    ax1.axvline(optimal_k, color='green', linestyle='--',
                label=f'k={optimal_k}')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Silhouette score', color='blue')
    ax2.set_ylabel('Inertia',          color='red')
    ax1.set_xticks(ks)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_pc_scatter(pc_scores, raw_labels, type_names, type_colors,
                    animal_id, output_path=None):
    """PC1 vs PC2 scatter, coloured by cluster."""
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(f'{animal_id} — PC scatter', fontweight='bold')
    for t, (name, col) in enumerate(zip(type_names, type_colors)):
        m = raw_labels == t
        ax.scatter(pc_scores[m, 0], pc_scores[m, 1],
                   c=col, alpha=0.5, s=15, label=f'{name} (n={m.sum()})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=7, markerscale=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_mean_profiles(mean_profiles, bin_centers, landmark_positions,
                        type_names, type_colors, animal_id, output_path=None):
    """Mean ± SEM spatial profile per cluster."""
    n_types = len(type_names)
    fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 3.5), sharey=True)
    fig.suptitle(f'{animal_id} — Mean profiles per cluster', fontweight='bold')
    if n_types == 1:
        axes = [axes]
    for t, ax in enumerate(axes):
        ax.plot(bin_centers, mean_profiles[t], color=type_colors[t], linewidth=2)
        for lp in landmark_positions:
            ax.axvline(lp, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_title(type_names[t], color=type_colors[t], fontweight='bold', fontsize=9)
        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Z-scored activity' if t == 0 else '')
        ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_global_proportions(raw_labels, layer_labels, layer_list,
                             type_names, type_colors, animal_id, output_path=None):
    """Stacked bar: cell type proportions per layer (all sessions pooled)."""
    n_types  = len(type_names)
    n_layers = len(layer_list)
    props  = np.zeros((n_types, n_layers))
    ns     = np.zeros(n_layers, dtype=int)
    for li, layer in enumerate(layer_list):
        mask = layer_labels == layer
        n    = int(np.sum(mask))
        ns[li] = n
        for t in range(n_types):
            props[t, li] = np.sum(raw_labels[mask] == t) / n if n > 0 else 0

    fig, ax = plt.subplots(figsize=(max(4, n_layers * 1.8), 4))
    fig.suptitle(f'{animal_id} — Cell type proportions by layer (pooled)',
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
    ax.set_xticklabels([f'{l}' for l in layer_list])
    ax.set_ylim(0, 120)
    ax.set_ylabel('% cells')
    handles = [mpatches.Patch(color=type_colors[t], label=type_names[t])
               for t in range(n_types)]
    ax.legend(handles=handles, fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_stacked_bars_by_layer(prop, counts, layer_list, session_order,
                                type_names, type_colors, animal_id, output_path=None):
    """Stacked bar per layer, session on x-axis."""
    n_layers = len(layer_list)
    n_types  = len(type_names)
    fig, axes = plt.subplots(1, n_layers, figsize=(3.5 * n_layers, 4), sharey=True)
    fig.suptitle(f'{animal_id} — Cell type proportions per layer × session',
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
        ax.set_ylabel('% cells' if li == 0 else '')
        ax.grid(True, alpha=0.2, axis='y')
    handles = [mpatches.Patch(color=type_colors[t], label=type_names[t])
               for t in range(n_types)]
    axes[-1].legend(handles=handles, fontsize=7, loc='upper right')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def _session_days(session_order):
    """Extract numeric day indices from session labels."""
    return [int(s.replace('Day', '')) for s in session_order]


def plot_trajectories_by_layer(prop, layer_list, session_order,
                                type_names, type_colors, animal_id, output_path=None):
    """Line plot: type proportions over days, one panel per layer."""
    n_layers = len(layer_list)
    n_types  = len(type_names)
    x = np.array(_session_days(session_order))
    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4), sharey=True)
    fig.suptitle(f'{animal_id} — Cell type trajectories per layer', fontweight='bold')
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
        ax.set_ylabel('% cells' if li == 0 else '')
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(session_order, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_trajectories_by_type(prop, layer_list, session_order,
                               type_names, type_colors, animal_id, output_path=None):
    """Line plot: layer trajectories, one panel per type."""
    n_types = len(type_names)
    x = np.array(_session_days(session_order))
    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4), sharey=True)
    fig.suptitle(f'{animal_id} — Layer trajectories per cell type', fontweight='bold')
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
        ax.set_ylabel('% cells' if t == 0 else '')
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(session_order, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


# ── Statistical tests ─────────────────────────────────────────

def run_chi_square_tests(raw_labels, session_labels, layer_labels,
                          session_order, layer_list, n_types):
    """Chi-square test: layer × cell type distribution, per session."""
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


def run_kruskal_layer_tests(raw_labels, layer_labels, layer_list, n_types,
                             type_names):
    """
    Kruskal-Wallis test per cell type: is the probability of belonging to type t
    significantly different across layers (pooled across all sessions)?
    """
    print("\n── Kruskal-Wallis: layer effect on cell type proportion ──")
    for t, tname in enumerate(type_names):
        groups = []
        for layer in layer_list:
            mask = layer_labels == layer
            if np.sum(mask) == 0:
                continue
            # Binary membership: 1 if cell is type t
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


# ── Save results ──────────────────────────────────────────────

def save_results_to_h5(filepath, raw_labels, type_names, pca, kmeans):
    """
    Persist cluster labels, type names, PCA components, and KMeans centroids
    back to the *_pca_data.h5 file so CellType_Evolution.py can reload the
    fitted model without re-fitting.
    """
    import pickle, io

    print(f"\n  Writing results to: {filepath}")
    with h5py.File(filepath, 'a') as f:
        # Clear old results
        for grp in ('pca_results', 'clustering'):
            if grp in f:
                del f[grp]

        # PCA results
        pr = f.create_group('pca_results')
        pr.create_dataset('components',              data=pca.components_)
        pr.create_dataset('explained_variance_ratio',data=pca.explained_variance_ratio_)
        pr.create_dataset('mean_',                   data=pca.mean_)
        pr.create_dataset('pc_scores',
                          data=pca.transform(
                              f['features/spatial_profiles_zscore'][:]))

        # Clustering
        cl = f.create_group('clustering')
        cl.create_dataset('raw_labels',          data=raw_labels)
        cl.create_dataset('kmeans_centers',      data=kmeans.cluster_centers_)
        cl.attrs['n_types'] = kmeans.n_clusters
        # Store type names as fixed-length strings
        cl.create_dataset('type_names',
                          data=np.array(type_names, dtype='S40'))

    print("  Done.")


# ── Main ──────────────────────────────────────────────────────

if __name__ == '__main__':

    out_dir = OUTPUT_DIR or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────
    print("=== 1. Load data ===")
    profiles, bin_centers, landmark_positions, \
        session_labels, layer_labels, session_order, animal_id = \
        load_pca_data(PCA_DATA_FILE)

    layer_list = [l for l in LAYER_ORDER
                  if np.any(layer_labels == l)]

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

    # ── 5. Population-level figures ───────────────────────────
    print("\n=== 5. Population figures ===")
    plot_scree(pca, N_CLUSTER_PCS, animal_id,
               os.path.join(out_dir, f'{animal_id}_scree.png'))

    plot_k_selection(sil_scores, inertias, K_RANGE, optimal_k, animal_id,
                     os.path.join(out_dir, f'{animal_id}_k_selection.png'))

    plot_pc_scatter(pc_scores, raw_labels, type_names, type_colors, animal_id,
                    os.path.join(out_dir, f'{animal_id}_pc_scatter.png'))

    plot_mean_profiles(mean_profiles, bin_centers, landmark_positions,
                       type_names, type_colors, animal_id,
                       os.path.join(out_dir, f'{animal_id}_mean_profiles.png'))

    plot_global_proportions(raw_labels, layer_labels, layer_list,
                             type_names, type_colors, animal_id,
                             os.path.join(out_dir, f'{animal_id}_global_proportions.png'))

    # ── 6. Layer × session proportion figures ─────────────────
    print("\n=== 6. Layer × session figures ===")
    prop, counts = build_prop_matrix(
        raw_labels, session_labels, layer_labels,
        session_order, layer_list, n_types
    )

    plot_stacked_bars_by_layer(
        prop, counts, layer_list, session_order,
        type_names, type_colors, animal_id,
        os.path.join(out_dir, f'{animal_id}_stacked_bars_by_layer.png')
    )

    plot_trajectories_by_layer(
        prop, layer_list, session_order,
        type_names, type_colors, animal_id,
        os.path.join(out_dir, f'{animal_id}_trajectories_by_layer.png')
    )

    plot_trajectories_by_type(
        prop, layer_list, session_order,
        type_names, type_colors, animal_id,
        os.path.join(out_dir, f'{animal_id}_trajectories_by_type.png')
    )

    # ── 7. Statistical tests ──────────────────────────────────
    print("\n=== 7. Statistical tests ===")
    run_chi_square_tests(raw_labels, session_labels, layer_labels,
                          session_order, layer_list, n_types)

    run_kruskal_layer_tests(raw_labels, layer_labels, layer_list,
                             n_types, type_names)

    # ── 8. Save results to HDF5 ───────────────────────────────
    print("\n=== 8. Save results ===")
    save_results_to_h5(PCA_DATA_FILE, raw_labels, type_names, pca, kmeans)

    print(f"\n=== Done — figures saved to: {out_dir} ===")
    plt.show()
