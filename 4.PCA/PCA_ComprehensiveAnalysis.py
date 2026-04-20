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
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, linregress
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# CONFIGURATION
# ============================================================

PCA_DATA_FILE    = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\PCA\JSY054_pca_data.h5"
# PCA_DATA_FILE    = r"F:\2P\unprocessed\JSY044\PCA\JSY044_pca_data.h5"
N_PCA_COMPONENTS = 10      # total PCs to fit
N_CLUSTER_PCS    = 5       # top PCs used for k-means
K_RANGE          = range(2, 8)

OVERRIDE_K       = None    # set to int to force a specific k

USE_ALIGNED_PROFILES =  False   # True  → use spatial_profiles_aligned (type-aware shifted)
                               # False → use spatial_profiles_zscore  (unshifted)

OUTPUT_DIR = os.path.join(os.path.dirname(PCA_DATA_FILE), "PCA_Figures_zscoreProfiles")  # None → same folder as this script

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
        # Profile selection priority:
        #   USE_ALIGNED_PROFILES=True  → aligned (type-aware shifted)
        #   USE_ALIGNED_PROFILES=False → session-corrected if present, else z-scored
        if USE_ALIGNED_PROFILES and 'features/spatial_profiles_aligned' in f:
            profiles = f['features/spatial_profiles_aligned'][:]
            print("  Using type-aware aligned profiles")
        elif 'features/spatial_profiles_session_corrected' in f:
            profiles = f['features/spatial_profiles_session_corrected'][:]
            print("  Using session-corrected profiles")
        else:
            profiles = f['features/spatial_profiles_zscore'][:]
            print("  Using z-scored profiles (unshifted)")

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

    names[_pick(np.argsort(peak_bins))]       = 'early-responding/adaptation-like'
    names[_pick(np.argsort(peak_bins)[::-1])] = 'LD4-preferring'
    # ranges = mean_profiles.max(axis=1) - mean_profiles.min(axis=1)
    # names[_pick(np.argsort(ranges))]           = 'Visually responsive'

    for k in range(n_types):
        if names[k] is None:
            peak_cm  = bin_centers[peak_bins[k]] if peak_bins[k] < len(bin_centers) else -1
            names[k] = f'Peak~{peak_cm:.0f}cm'

    # Cluster colours chosen to avoid layer palette (green, blue, orange, purple)
    base_colors = {
        'Adaptation-like':    '#C62828',  # deep red
        'L4-preferring':      '#00838F',  # teal
        'Visually responsive':'#F9A825',  # amber/goldenrod
    }
    extra = ['#6D4C41', '#AD1457', '#558B2F', '#00695C']  # brown, rose, olive, dark teal
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


# ── Statistical helpers ───────────────────────────────────────

def _sig_stars(p):
    return '***' if p < 0.001 else ('**' if p < 0.01
           else ('*' if p < 0.05 else 'ns'))


def _draw_sig_bracket(ax, x1, x2, y, h, text, fontsize=8):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c='black')
    ax.text((x1 + x2) / 2, y + h, text,
            ha='center', va='bottom', fontsize=fontsize)


def run_layer_posthoc(raw_labels, layer_labels, layer_list, n_types, type_names):
    """
    Per cell type: Kruskal-Wallis across layers (pooled sessions) +
    Bonferroni-corrected pairwise Mann-Whitney U post-hoc tests.
    Returns results[tname] = {'kw_p': float, 'posthoc': {(l1,l2): p_adj}}
    """
    print("\n── Layer effect: KW + post-hoc Mann-Whitney (Bonferroni) ──")
    results = {}
    for t, tname in enumerate(type_names):
        groups, grp_layers = [], []
        for layer in layer_list:
            mask = layer_labels == layer
            if np.sum(mask) == 0:
                continue
            groups.append((raw_labels[mask] == t).astype(float))
            grp_layers.append(layer)

        if len(groups) < 2:
            print(f"  {tname}: not enough layers")
            continue

        try:
            stat, kw_p = kruskal(*groups)
            sig = _sig_stars(kw_p)
            print(f"\n  {tname}: KW H={stat:.2f}  p={kw_p:.4f}  {sig}")
            for layer, grp in zip(grp_layers, groups):
                print(f"    {layer}: {np.mean(grp)*100:.1f}%  (n={len(grp)})")
        except Exception as e:
            print(f"  {tname}: KW failed ({e})")
            continue

        # Pairwise post-hoc
        pairs = [(i, j) for i in range(len(groups))
                         for j in range(i + 1, len(groups))]
        n_pairs = len(pairs)
        posthoc = {}
        for i, j in pairs:
            try:
                _, p_mw = mannwhitneyu(groups[i], groups[j],
                                       alternative='two-sided')
                p_adj = min(p_mw * n_pairs, 1.0)
                key   = (grp_layers[i], grp_layers[j])
                posthoc[key] = p_adj
                print(f"    {grp_layers[i]} vs {grp_layers[j]}: "
                      f"p_adj={p_adj:.4f}  {_sig_stars(p_adj)}")
            except Exception:
                pass

        results[tname] = {'kw_p': kw_p, 'posthoc': posthoc}
    return results


def plot_layer_posthoc(raw_labels, layer_labels, layer_list, n_types,
                        type_names, type_colors, animal_id, output_path=None):
    """
    Bar = proportion per layer (pooled sessions) per cell type,
    with Bonferroni-corrected Mann-Whitney U significance brackets.
    KW omnibus p shown in x-label.
    """
    fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 5))
    fig.suptitle(f'{animal_id} — Layer effect + post-hoc (Bonferroni)',
                 fontweight='bold')
    if n_types == 1:
        axes = [axes]

    xpos = np.arange(len(layer_list))
    for t, (tname, tcolor, ax) in enumerate(zip(type_names, type_colors, axes)):
        grand_mean, groups_here, grp_layers = [], [], []
        for layer in layer_list:
            mask = layer_labels == layer
            n    = int(np.sum(mask))
            grand_mean.append(np.mean(raw_labels[mask] == t) * 100 if n > 0 else 0)
            if n > 0:
                groups_here.append((raw_labels[mask] == t).astype(float))
                grp_layers.append(layer)

        ax.bar(xpos, grand_mean, color=tcolor, alpha=0.7, width=0.6, zorder=2)

        for xi, (layer, val) in enumerate(zip(layer_list, grand_mean)):
            n = int(np.sum(layer_labels == layer))
            ax.text(xi, val + 1, f'n={n}', ha='center', va='bottom',
                    fontsize=7, rotation=45)

        # Post-hoc brackets
        pairs   = [(i, j) for i in range(len(groups_here))
                           for j in range(i + 1, len(groups_here))]
        n_pairs = len(pairs)
        sig_pairs = []
        for i, j in pairs:
            try:
                _, p_mw = mannwhitneyu(groups_here[i], groups_here[j],
                                       alternative='two-sided')
                p_adj = min(p_mw * n_pairs, 1.0)
                if p_adj < 0.05:
                    xi = layer_list.index(grp_layers[i])
                    xj = layer_list.index(grp_layers[j])
                    sig_pairs.append((xi, xj, _sig_stars(p_adj)))
            except Exception:
                pass

        y_top = max(grand_mean) if grand_mean else 0
        step  = max(y_top * 0.13, 3)
        base  = y_top * 1.15
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

        try:
            _, kw_p = kruskal(*groups_here)
            ax.set_xlabel(f'KW p={kw_p:.4f} {_sig_stars(kw_p)}', fontsize=8)
        except Exception:
            pass

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def run_experience_effect(raw_labels, session_labels, layer_labels,
                           session_order, layer_list, n_types, type_names):
    """
    Per cell type × layer: OLS regression of proportion ~ recording day.
    Returns results[tname][layer] = {'slope': float, 'p': float, 'days': array,
                                      'props': array}
    """
    print("\n── Experience effect: proportion ~ day (OLS per type × layer) ──")
    days = np.array([int(s.replace('Day', '')) for s in session_order], dtype=float)
    results = {}
    for t, tname in enumerate(type_names):
        print(f"\n  {tname}:")
        results[tname] = {}
        for layer in layer_list:
            props, valid_days = [], []
            for si, sess in enumerate(session_order):
                mask = (session_labels == sess) & (layer_labels == layer)
                n    = int(np.sum(mask))
                if n == 0:
                    continue
                props.append(np.mean(raw_labels[mask] == t) * 100)
                valid_days.append(days[si])

            if len(valid_days) < 3:
                results[tname][layer] = None
                continue

            vd = np.array(valid_days)
            vp = np.array(props)
            try:
                slope, intercept, r, p, _ = linregress(vd, vp)
                print(f"    {layer}: slope={slope:+.2f}%/day  "
                      f"r={r:.3f}  p={p:.4f}  {_sig_stars(p)}")
                results[tname][layer] = {
                    'slope': slope, 'intercept': intercept,
                    'r': r, 'p': p,
                    'days': vd, 'props': vp,
                }
            except Exception as e:
                print(f"    {layer}: failed ({e})")
                results[tname][layer] = None
    return results


def plot_experience_regression(exp_results, layer_list, session_order,
                                type_names, type_colors, animal_id,
                                output_path=None):
    """
    Grid: rows = cell types, columns = layers.
    Each panel: scatter of proportion per session + OLS line.
    Solid line = p<0.05, dashed = ns.  Slope + p annotated.
    """
    n_types  = len(type_names)
    n_layers = len(layer_list)
    days_all = np.array([int(s.replace('Day', '')) for s in session_order],
                        dtype=float)

    fig, axes = plt.subplots(n_types, n_layers,
                             figsize=(4 * n_layers, 3.5 * n_types),
                             sharey='row', sharex='col')
    fig.suptitle(f'{animal_id} — Experience effect: proportion ~ day',
                 fontweight='bold')

    # Ensure 2-D array
    if n_types == 1:
        axes = axes[np.newaxis, :]
    if n_layers == 1:
        axes = axes[:, np.newaxis]

    for t, tname in enumerate(type_names):
        for li, layer in enumerate(layer_list):
            ax  = axes[t, li]
            res = exp_results[tname].get(layer)

            if t == 0:
                ax.set_title(layer,
                             color=LAYER_COLORS.get(layer, 'black'),
                             fontweight='bold')
            if li == 0:
                ax.set_ylabel(f'{tname}\n% cells',
                              color=type_colors[t], fontsize=8)

            if res is None:
                ax.text(0.5, 0.5, 'insufficient data',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=7, color='gray')
                ax.set_xlabel('Day' if t == n_types - 1 else '')
                ax.grid(True, alpha=0.2)
                continue

            ax.scatter(res['days'], res['props'],
                       color=type_colors[t], s=40, zorder=3)

            x_line = np.array([res['days'].min(), res['days'].max()])
            y_line = res['intercept'] + res['slope'] * x_line
            ls = '-' if res['p'] < 0.05 else '--'
            ax.plot(x_line, y_line, color=type_colors[t],
                    linewidth=1.8, linestyle=ls, zorder=2)

            ax.annotate(f"slope={res['slope']:+.2f}%/day\n"
                        f"p={res['p']:.3f} {_sig_stars(res['p'])}",
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=7, va='top',
                        color='black')

            ax.set_xticks(res['days'])
            ax.set_xticklabels(
                [s for s, d in zip(session_order, days_all)
                 if d in res['days']],
                rotation=45, ha='right', fontsize=7)
            ax.set_xlabel('Day' if t == n_types - 1 else '')
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def run_session_trend_tests(raw_labels, session_labels, layer_labels,
                             session_order, layer_list, n_types, type_names,
                             n_early=3, n_late=3):
    """
    Per cell type × layer:
      1. Kruskal-Wallis across all sessions — omnibus "did anything change?"
      2. Early (first n_early sessions) vs late (last n_late sessions)
         Mann-Whitney U — net direction of change.

    Returns results[tname][layer] = {
        'kw_p': float, 'mw_p': float, 'direction': str,
        'early_mean': float, 'late_mean': float,
        'days': array, 'props': array }
    """
    print(f"\n── Session trend: KW (omnibus) + early({n_early}) vs "
          f"late({n_late}) MW-U ──")
    results = {}

    for t, tname in enumerate(type_names):
        print(f"\n  {tname}:")
        results[tname] = {}

        for layer in layer_list:
            # Collect per-session proportions
            props, valid_days = [], []
            days_all = np.array([int(s.replace('Day', ''))
                                 for s in session_order], dtype=float)
            for si, sess in enumerate(session_order):
                mask = (session_labels == sess) & (layer_labels == layer)
                n    = int(np.sum(mask))
                if n == 0:
                    continue
                props.append(np.mean(raw_labels[mask] == t) * 100)
                valid_days.append(days_all[si])

            if len(props) < 3:
                results[tname][layer] = None
                continue

            props      = np.array(props)
            valid_days = np.array(valid_days)

            # 1. KW across sessions (each session is one "group" = binary
            #    membership values for all cells in that session × layer)
            kw_groups = []
            for si, sess in enumerate(session_order):
                mask = (session_labels == sess) & (layer_labels == layer)
                if np.sum(mask) == 0:
                    continue
                kw_groups.append((raw_labels[mask] == t).astype(float))

            kw_p = np.nan
            if len(kw_groups) >= 2:
                try:
                    _, kw_p = kruskal(*kw_groups)
                except Exception:
                    pass

            # 2. Early vs late MW-U
            early_vals = props[:n_early]
            late_vals  = props[len(props) - n_late:]
            mw_p, direction = np.nan, 'n/a'
            if len(early_vals) >= 1 and len(late_vals) >= 1:
                try:
                    _, mw_p = mannwhitneyu(early_vals, late_vals,
                                           alternative='two-sided')
                    direction = ('increase' if np.mean(late_vals) > np.mean(early_vals)
                                 else 'decrease')
                except Exception:
                    pass

            print(f"    {layer}: KW p={kw_p:.4f} {_sig_stars(kw_p)}  |  "
                  f"early={np.mean(early_vals):.1f}% → late={np.mean(late_vals):.1f}%  "
                  f"MW p={mw_p:.4f} {_sig_stars(mw_p)}  [{direction}]")

            results[tname][layer] = {
                'kw_p':       kw_p,
                'mw_p':       mw_p,
                'direction':  direction,
                'early_mean': float(np.mean(early_vals)),
                'late_mean':  float(np.mean(late_vals)),
                'days':       valid_days,
                'props':      props,
            }

    return results


def plot_session_trend(trend_results, layer_list, session_order,
                        type_names, type_colors, animal_id,
                        n_early=2, n_late=2, output_path=None):
    """
    Grid: rows = cell types, columns = layers.
    Each panel: scatter of proportion per session with early/late shading,
    KW p + early-vs-late MW-U p annotated.
    """
    n_types  = len(type_names)
    n_layers = len(layer_list)

    fig, axes = plt.subplots(n_types, n_layers,
                             figsize=(4 * n_layers, 3.5 * n_types),
                             sharey='row', sharex='col')
    fig.suptitle(f'{animal_id} — Session trend: KW omnibus + early vs late',
                 fontweight='bold')

    if n_types == 1:
        axes = axes[np.newaxis, :]
    if n_layers == 1:
        axes = axes[:, np.newaxis]

    for t, tname in enumerate(type_names):
        for li, layer in enumerate(layer_list):
            ax  = axes[t, li]
            res = trend_results[tname].get(layer)

            if t == 0:
                ax.set_title(layer,
                             color=LAYER_COLORS.get(layer, 'black'),
                             fontweight='bold')
            if li == 0:
                ax.set_ylabel(f'{tname}\n% cells',
                              color=type_colors[t], fontsize=8)

            if res is None:
                ax.text(0.5, 0.5, 'insufficient data',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=7, color='gray')
                ax.set_xlabel('Session' if t == n_types - 1 else '')
                ax.grid(True, alpha=0.2)
                continue

            days  = res['days']
            props = res['props']
            xpos  = np.arange(len(days))

            # Shade early / late
            if len(xpos) >= n_early:
                ax.axvspan(-0.5, n_early - 0.5, color='skyblue',
                           alpha=0.15, label='early')
            if len(xpos) >= n_late:
                ax.axvspan(len(xpos) - n_late - 0.5, len(xpos) - 0.5,
                           color='salmon', alpha=0.15, label='late')

            ax.scatter(xpos, props, color=type_colors[t], s=45, zorder=3)
            ax.plot(xpos, props, color=type_colors[t],
                    linewidth=1.2, alpha=0.6, zorder=2)

            # Early / late mean lines
            ax.hlines(res['early_mean'], -0.5, n_early - 0.5,
                      colors='steelblue', linewidths=1.5, linestyles='--')
            ax.hlines(res['late_mean'], len(xpos) - n_late - 0.5, len(xpos) - 0.5,
                      colors='firebrick', linewidths=1.5, linestyles='--')

            kw_str = f"KW p={res['kw_p']:.3f} {_sig_stars(res['kw_p'])}"
            mw_str = (f"early→late: {res['direction']}\n"
                      f"MW p={res['mw_p']:.3f} {_sig_stars(res['mw_p'])}")
            ax.annotate(f"{kw_str}\n{mw_str}",
                        xy=(0.03, 0.97), xycoords='axes fraction',
                        fontsize=6.5, va='top', color='black')

            ax.set_xticks(xpos)
            ax.set_xticklabels(session_order[:len(xpos)],
                               rotation=45, ha='right', fontsize=7)
            ax.set_xlabel('Session' if t == n_types - 1 else '')
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


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


# ── Cluster response heatmaps (unaligned profiles) ───────────

def plot_cluster_response_heatmaps(filepath, raw_labels, session_labels,
                                    layer_labels, bin_centers, landmark_positions,
                                    type_names, type_colors, session_order,
                                    layer_list, animal_id, output_dir=None):
    """
    For each cluster, produce one figure: rows = layers, columns = sessions.
    Each panel is a heatmap of the RAW z-scored (unaligned) spatial profiles
    for cells in that cluster, sorted by peak position within the panel.

    Profiles are always loaded from spatial_profiles_zscore (unaligned)
    regardless of USE_ALIGNED_PROFILES, so you can visually verify cluster
    assignments against the raw responses.

    Landmark positions are marked as dashed vertical lines.
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Always load unaligned z-scored profiles
    with h5py.File(filepath, 'r') as f:
        unaligned = f['features/spatial_profiles_zscore'][:]

    cmap = LinearSegmentedColormap.from_list(
        'WBR', [(0.2, 0.2, 0.8), (1, 1, 1), (0.8, 0.1, 0.1)])

    n_sessions = len(session_order)
    n_layers   = len(layer_list)

    for t, (tname, tcolor) in enumerate(zip(type_names, type_colors)):
        tmask = raw_labels == t

        fig, axes = plt.subplots(
            n_layers, n_sessions,
            figsize=(2.5 * n_sessions, 2.8 * n_layers),
            squeeze=False)
        fig.suptitle(
            f'{animal_id} — {tname} (n={int(np.sum(tmask))}) — unaligned z-scored profiles',
            fontweight='bold', color=tcolor, fontsize=10)

        for li, layer in enumerate(layer_list):
            for si, sess in enumerate(session_order):
                ax = axes[li, si]

                mask = tmask & (session_labels == sess) & (layer_labels == layer)
                n    = int(np.sum(mask))

                if n == 0:
                    ax.set_visible(False)
                    continue

                data = unaligned[mask]  # (n, n_bins)

                # Normalise each cell 0→1 for display (matches create_response_plot)
                data_norm = data.copy()
                for i in range(len(data_norm)):
                    mn, mx = data_norm[i].min(), data_norm[i].max()
                    if mx > mn:
                        data_norm[i] = (data_norm[i] - mn) / (mx - mn)

                # Sort by peak position
                sort_idx  = np.argsort(np.argmax(data_norm, axis=1))
                data_sort = data_norm[sort_idx]

                vmax = np.percentile(data_sort, 95)
                ax.imshow(data_sort, aspect='auto', cmap=cmap,
                                 vmin=0, vmax=max(vmax, 0.1),
                                 interpolation='nearest',
                                 extent=[bin_centers[0], bin_centers[-1], n, 0])

                # Landmark lines
                for lm in landmark_positions:
                    ax.axvline(lm, color='k', linestyle='--', linewidth=0.8, alpha=0.6)

                ax.set_title(f'{sess}  n={n}', fontsize=7)
                if si == 0:
                    ax.set_ylabel(layer, fontsize=8,
                                  color=LAYER_COLORS.get(layer, 'black'))
                else:
                    ax.set_yticks([])
                if li == n_layers - 1:
                    ax.set_xlabel('Position (cm)', fontsize=7)
                else:
                    ax.set_xticks([])

        plt.tight_layout()
        fname = f'{animal_id}_{tname.replace(" ", "_").replace("/", "-")}_unaligned_heatmap.png'
        if output_dir:
            fpath = os.path.join(output_dir, fname)
            fig.savefig(fpath, dpi=150, bbox_inches='tight')
            print(f"  Saved: {fname}")

    return


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

    layer_posthoc_results = run_layer_posthoc(
        raw_labels, layer_labels, layer_list, n_types, type_names)

    exp_results = run_experience_effect(
        raw_labels, session_labels, layer_labels,
        session_order, layer_list, n_types, type_names)

    # ── 8. Statistical figures ────────────────────────────────
    print("\n=== 8. Statistical figures ===")
    plot_layer_posthoc(
        raw_labels, layer_labels, layer_list, n_types,
        type_names, type_colors, animal_id,
        os.path.join(out_dir, f'{animal_id}_layer_posthoc.png'))

    plot_experience_regression(
        exp_results, layer_list, session_order,
        type_names, type_colors, animal_id,
        os.path.join(out_dir, f'{animal_id}_experience_regression.png'))

    trend_results = run_session_trend_tests(
        raw_labels, session_labels, layer_labels,
        session_order, layer_list, n_types, type_names)

    plot_session_trend(
        trend_results, layer_list, session_order,
        type_names, type_colors, animal_id,
        output_path=os.path.join(out_dir, f'{animal_id}_session_trend.png'))

    # ── 9. Cluster response heatmaps (unaligned) ──────────────
    print("\n=== 9. Cluster response heatmaps ===")
    plot_cluster_response_heatmaps(
        PCA_DATA_FILE, raw_labels, session_labels, layer_labels,
        bin_centers, landmark_positions, type_names, type_colors,
        session_order, layer_list, animal_id, output_dir=out_dir)

    # ── 10. Save results to HDF5 ──────────────────────────────
    print("\n=== 10. Save results ===")
    save_results_to_h5(PCA_DATA_FILE, raw_labels, type_names, pca, kmeans)

    print(f"\n=== Done — figures saved to: {out_dir} ===")
    plt.show()
