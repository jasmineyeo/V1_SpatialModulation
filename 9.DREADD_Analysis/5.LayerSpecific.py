"""
Phase 5 -- Layer-Specific SMI Effects (DREADD saline/DCZ cohort)
==================================================================

Consolidated from 5.LayerSpecific.ipynb once every function below worked
end-to-end on real JSY090 and JSY093 data.

Scope
-----
Track B only for now, same reasoning as Phase 4 -- Track A's layer-specific
mixed-effects analysis is deferred until Track A gets revisited across
Phases 3-6 after this phase and Phase 6 are done for Track B. This is
being tracked deliberately, not forgotten -- same parking convention as
Phase 3/4's Track A note.

Tests the deep-layer sensitivity hypothesis: since RSC projects directly
to V1's deep layers (L5/L6), DCZ's effect on spatial coding (if any) was
predicted to be stronger in L5/L6 than in superficial layers (L2/3, L4).

Loads Phase 4's already-saved comparison-group tables directly
(`{group}_comparison_table.csv`, from Function 4.8) rather than
re-deriving `all_group_dfs` via the interactive picker -- the whole point
of saving those tables was so later phases wouldn't need to redo that.

Findings so far (both animals, Track B) -- NOT conclusive, see caveats
------------------------------------------------------------------------
- JSY090: no detectable DCZ effect anywhere -- population SMI, any single
  layer, any pooled depth group, any comparison group. Consistent with
  Phase 4's finding that this animal is a likely weak/non-responder;
  nothing to decompose by layer here.
- JSY093 (the animal with a real overall DCZ effect from Phase 4): the
  formal deep-vs-superficial interaction test (rank(SMI) ~ condition *
  depth_group) never reached significance in any of the 5 comparison
  groups -- no statistical support for the deep-layer-specificity
  hypothesis as originally framed.
- Looking at the pooled deep/superficial comparisons directly (not just
  the interaction test) initially looked like a "broadly distributed,
  not deep-specific" story -- effect sizes were comparable in both pooled
  buckets in the best-powered groups (DCZ1: -0.142 deep vs -0.142
  superficial; DCZ2: -0.249 deep vs -0.216 superficial).
- BUT the per-layer (unpooled) picture is patchier than either "deep
  specific" or "uniformly distributed" -- e.g. in DCZ2, L5 shows an
  essentially zero dcz-vs-saline effect (+0.006, p=0.55) while L6 (also
  "deep") shows the largest effect of any layer in that group (-0.324,
  p=0.0047). L4 (the canonical thalamic input layer, not usually a target
  of cortico-cortical feedback like RSC->V1) is NOT spared -- it shows one
  of the largest, most significant effects in both DCZ1 and DCZ2, the two
  best-powered groups. This argues against a clean "RSC only reaches its
  direct anatomical targets" story, but the pattern (L5 silent next to L6
  in DCZ2, L4 affected despite not being an expected direct target) isn't
  yet interpretable as a clean alternative story either.
- Active_OL and Stationary_OL have very thin per-layer saline N in
  several layers (see Function 5.9's warnings) -- their non-significant
  per-layer/per-depth results are more likely underpowered than genuinely
  null; don't read too much into them either way.
- Bottom line: this phase's data cannot yet support a firm claim about
  WHICH layers are functionally affected by DCZ, in either direction. A
  whole-session population SMI measure may simply be too coarse to
  resolve this -- Track A's per-cell resolution, and/or more animals, may
  be needed before this settles into an interpretable pattern.

Correction (added later, see Functions 5.11-5.13): the tests above
(5.1/5.2's Kruskal-Wallis/Mann-Whitney, 5.4's rank-ANCOVA interaction)
all pool CELLS within a layer as the unit of comparison. For any single
comparison group (e.g. DCZ1), that's one saline session and one paired
dcz session -- so cells within it are not independent replicates of that
condition, the same pseudo-replication problem later caught and fixed in
Phase 6's landmark-preference work. Functions 5.11-5.13 redo the
per-layer saline-vs-dcz question and a new layer-heterogeneity question
("does the SIZE of the effect differ significantly between layers?",
never actually tested before) using each comparison GROUP as the
independent unit (paired t-test / repeated-measures ANOVA across the 5
groups, not across pooled cells) -- the same fix Phase 6 applied. Results
pending a real-data run; this note will be updated once that's done.

Indexing / reuse convention (shared with Phases 0-4)
------------------------------------------------------
Every per-cell array here is indexed by POSITION in
`stat[iscell[:, 0] == 1]`, same as every earlier phase. No index
translation is ever needed between this phase's inputs (Phase 4's saved
`{group}_comparison_table.csv`) and its outputs.
"""

import os
import glob
import json
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # interactive popups need a real GUI backend
import matplotlib.pyplot as plt
from matplotlib import rcParams
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM
from scipy.stats import kruskal, mannwhitneyu, rankdata, wilcoxon, ttest_rel

rcParams['legend.fontsize'] = 40
rcParams['axes.labelsize'] = 40
rcParams['axes.titlesize'] = 50
rcParams['xtick.labelsize'] = 40
rcParams['ytick.labelsize'] = 40

# rcParams['legend.fontsize'] = 20
# rcParams['axes.labelsize'] = 20
# rcParams['axes.titlesize'] = 25
# rcParams['xtick.labelsize'] = 20
# rcParams['ytick.labelsize'] = 20


# =============================================================================
# Setup -- load Phase 4's already-saved comparison tables
# =============================================================================

def load_all_group_dfs_from_phase4(output_dir):
    """
    Load every comparison group's table already saved by Phase 4's
    Function 4.8 (save_all_phase4_outputs) -- Phase 4 already saved these
    to '{group_name}_comparison_table.csv' per group, so Phase 5 doesn't
    need to redo the interactive picker at all.

    Parameters
    ----------
    output_dir : str
        e.g. os.path.join(ANIMAL_DIR, 'Phase4_SessionComparison_Results').

    Returns
    -------
    all_group_dfs : dict
        {group_name: df}.
    """
    csv_paths = sorted(glob.glob(os.path.join(output_dir, '*_comparison_table.csv')))

    if not csv_paths:
        raise FileNotFoundError(f"No *_comparison_table.csv found in {output_dir} -- "
                                 "has Phase 4's save step (Function 4.8) been run for this animal?")

    all_group_dfs = {}
    for csv_path in csv_paths:
        group_name = os.path.basename(csv_path)[:-len('_comparison_table.csv')]
        df = pd.read_csv(csv_path)
        all_group_dfs[group_name] = df
        print(f"Loaded '{group_name}': {len(df)} cell-rows <- {csv_path}")

    print(f"\nLoaded {len(all_group_dfs)} group(s) from {output_dir}: {list(all_group_dfs.keys())}")
    return all_group_dfs


# =============================================================================
# Function 5.1 -- compare_smi_across_conditions
#
# Reimplemented from Phase 4's Function 4.7, unchanged -- the core
# Kruskal-Wallis omnibus + pairwise Mann-Whitney U (Holm-corrected within
# the group's own pairwise set) logic, reused by every layer-level test
# below.
# =============================================================================

def compare_smi_across_conditions(df, group_col='condition', value_col='SMI', filter_col='valid'):
    """
    Kruskal-Wallis omnibus + pairwise Mann-Whitney U (Holm-corrected)
    across whatever categories are present.

    Parameters
    ----------
    df : pandas.DataFrame
        One group's or one layer's/depth-group's table.
    group_col, value_col, filter_col : str

    Returns
    -------
    result : dict or None
        None (with a printed message) if fewer than 2 categories remain
        after filtering. Otherwise:
        {'group_medians': {cat: median}, 'group_n': {cat: n},
         'omnibus_stat', 'omnibus_p',
         'pairwise': DataFrame(cond_a, cond_b, median_diff, U_stat, p_raw, p_holm)}.
    """
    filtered = df[df[filter_col]]
    categories = [c for c in filtered[group_col].unique() if pd.notna(c)]

    if len(categories) < 2:
        print(f"Only {len(categories)} category(ies) present after filtering on '{filter_col}' "
              f"-- nothing to compare ({categories}).")
        return None

    samples = {cat: filtered.loc[filtered[group_col] == cat, value_col].to_numpy()
               for cat in categories}

    group_medians = {cat: float(np.median(vals)) for cat, vals in samples.items()}
    group_n = {cat: len(vals) for cat, vals in samples.items()}

    omnibus_stat, omnibus_p = kruskal(*samples.values())

    pairwise_rows = []
    for cat_a, cat_b in combinations(categories, 2):
        u_stat, p_raw = mannwhitneyu(samples[cat_a], samples[cat_b], alternative='two-sided')
        pairwise_rows.append({
            'cond_a': cat_a, 'cond_b': cat_b,
            'median_diff': group_medians[cat_a] - group_medians[cat_b],
            'U_stat': u_stat, 'p_raw': p_raw,
        })

    pairwise_df = pd.DataFrame(pairwise_rows)
    if len(pairwise_df) > 0:
        _, p_holm, _, _ = multipletests(pairwise_df['p_raw'], method='holm')
        pairwise_df['p_holm'] = p_holm

    print(f"Categories ({group_col}): {categories}")
    print(f"  n per category: {group_n}")
    print(f"  median {value_col} per category: {group_medians}")
    print(f"  Kruskal-Wallis: H={omnibus_stat:.3f}, p={omnibus_p:.4f}")
    print(f"\n  Pairwise (Holm-corrected):")
    print(pairwise_df.to_string(index=False))

    return {
        'group_medians': group_medians,
        'group_n': group_n,
        'omnibus_stat': omnibus_stat,
        'omnibus_p': omnibus_p,
        'pairwise': pairwise_df,
    }


# =============================================================================
# Function 5.2 -- compare_smi_by_layer
#
# Loops Function 5.1 over each layer present (L2/3, L4, L5, L6, in that
# canonical order) in one group's table. Caveat: each layer's Holm
# correction is applied only within that layer's own ~3 pairwise tests,
# not across all layers x pairs combined for a group.
# =============================================================================

CANONICAL_LAYER_ORDER = ['L2/3', 'L4', 'L5', 'L6']


def _layer_order(layers_present):
    return ([l for l in CANONICAL_LAYER_ORDER if l in layers_present]
            + [l for l in layers_present if l not in CANONICAL_LAYER_ORDER])


def compare_smi_by_layer(df, layer_col='layer', group_col='condition', value_col='SMI', filter_col='valid'):
    """
    Loop compare_smi_across_conditions over each layer present.

    Parameters
    ----------
    df : pandas.DataFrame
        One group's table.
    layer_col, group_col, value_col, filter_col : str

    Returns
    -------
    results : dict
        {layer: compare_smi_across_conditions(...) result or None}.
    summary_df : pandas.DataFrame
        One row per (layer, pairwise comparison) that had 2+ categories.
    """
    layers_present = [l for l in df[layer_col].dropna().unique()]
    layer_order = _layer_order(layers_present)

    results = {}
    summary_rows = []
    for layer in layer_order:
        print(f"\n--- Layer {layer} ---")
        layer_df = df[df[layer_col] == layer]
        result = compare_smi_across_conditions(layer_df, group_col=group_col, value_col=value_col,
                                                filter_col=filter_col)
        results[layer] = result
        if result is not None:
            for _, row in result['pairwise'].iterrows():
                summary_rows.append({
                    'layer': layer, 'cond_a': row['cond_a'], 'cond_b': row['cond_b'],
                    'median_diff': row['median_diff'], 'p_holm': row['p_holm'],
                })

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) > 0:
        print("\n=== Summary across layers (each layer's own Holm correction -- not corrected across layers) ===")
        print(summary_df.to_string(index=False))

    return results, summary_df


# =============================================================================
# Function 5.3 -- plot_smi_by_layer
# =============================================================================

def plot_smi_by_layer(df, layer_col='layer', group_col='condition', value_col='SMI', filter_col='valid', title=''):
    """
    Grid of violin+strip plots, one panel per layer present.

    Parameters
    ----------
    df : pandas.DataFrame
    layer_col, group_col, value_col, filter_col : str
    title : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    filtered = df[df[filter_col]]
    layers_present = [l for l in filtered[layer_col].dropna().unique()]
    layer_order = _layer_order(layers_present)

    color_by_category = {'baseline': 'tab:blue', 'saline': 'tab:orange', 'dcz': 'tab:green'}

    n_layers = len(layer_order)
    n_cols = 2
    n_rows = int(np.ceil(n_layers / n_cols)) if n_layers > 0 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, layer in zip(axes, layer_order):
        layer_df = filtered[filtered[layer_col] == layer]
        category_order = [c for c in ['baseline', 'saline', 'dcz'] if c in layer_df[group_col].unique()]
        category_order += [c for c in layer_df[group_col].unique() if c not in category_order]

        data_by_category = [layer_df.loc[layer_df[group_col] == cat, value_col].to_numpy()
                            for cat in category_order]

        if len(data_by_category) == 0 or all(len(d) == 0 for d in data_by_category):
            ax.set_title(f"{layer} (no data)")
            ax.axis('off')
            continue

        parts = ax.violinplot(data_by_category, showmedians=True)
        for i, body in enumerate(parts['bodies']):
            body.set_facecolor(color_by_category.get(category_order[i], 'gray'))
            body.set_alpha(0.4)

        rng = np.random.default_rng(0)
        for i, vals in enumerate(data_by_category):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                       color=color_by_category.get(category_order[i], 'gray'), s=12, alpha=0.5)

        ax.set_ylim(-1.3, 1.3)
        ax.set_yticks(np.arange(-1, 1.1, 0.5))
        ax.set_xticks(range(1, len(category_order) + 1))
        ax.set_xticklabels(category_order)
        ax.set_ylabel(value_col)
        ax.set_title(layer)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    for ax in axes[len(layer_order):]:
        ax.axis('off')

    fig.suptitle(title, fontsize=20, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# Function 5.4 -- test_layer_depth_interaction
#
# The formal test of the layer-specificity hypothesis: collapses layers
# into depth_group (L2/3+L4 = superficial, L5+L6 = deep), rank-transforms
# SMI (robust to non-normality, consistent with the non-parametric choice
# in Function 4.7/5.1), and fits rank(SMI) ~ C(condition) * C(depth_group).
# The interaction term tests whether the condition effect's SIZE differs
# between deep and superficial -- never reached significance in any group,
# either animal (see module docstring).
# =============================================================================

def test_layer_depth_interaction(df, layer_col='layer', group_col='condition', value_col='SMI', filter_col='valid',
                                  deep_layers=('L5', 'L6'), superficial_layers=('L2/3', 'L4')):
    """
    rank(SMI) ~ C(condition) * C(depth_group).

    Parameters
    ----------
    df : pandas.DataFrame
        One group's table.
    layer_col, group_col, value_col, filter_col : str
    deep_layers, superficial_layers : tuple of str

    Returns
    -------
    model_result : statsmodels regression results object, or None
        (fewer than 2 conditions or depth groups present).
    """
    filtered = df[df[filter_col]].copy()

    def _depth(layer):
        if layer in deep_layers:
            return 'deep'
        elif layer in superficial_layers:
            return 'superficial'
        return None

    filtered['depth_group'] = filtered[layer_col].map(_depth)
    filtered = filtered.dropna(subset=['depth_group', group_col, value_col])

    conditions_present = filtered[group_col].unique()
    depths_present = filtered['depth_group'].unique()

    if len(conditions_present) < 2 or len(depths_present) < 2:
        print(f"Not enough categories to test an interaction (conditions={list(conditions_present)}, "
              f"depths={list(depths_present)}) -- skipping.")
        return None

    filtered['SMI_rank'] = rankdata(filtered[value_col])

    formula = f"SMI_rank ~ C({group_col}) * C(depth_group)"
    model_result = smf.ols(formula, data=filtered).fit()

    print(f"\n=== Layer-depth interaction: rank({value_col}) ~ {group_col} * depth_group ===")
    print(model_result.summary().tables[1])

    interaction_terms = [p for p in model_result.params.index if ':' in p]
    if interaction_terms:
        print(f"\nInteraction term(s): {interaction_terms}")
        print("(a significant interaction term means the condition effect's SIZE differs "
              "between deep and superficial layers)")

    return model_result


# =============================================================================
# Function 5.7 -- compare_smi_by_depth_group
#
# Same core comparison as Function 5.2, but pooling layers into just two
# buckets first (deep = L5+L6, superficial = L2/3+L4). More cells per
# bucket than any single layer, so more power than the per-layer
# breakdown -- a direct complement to Function 5.4's interaction p-value.
# Numbered 5.7 (not 5.5/5.6, which are the drivers below) to match the
# order these were actually added in.
# =============================================================================

def _assign_depth_group(layer_series, deep_layers=('L5', 'L6'), superficial_layers=('L2/3', 'L4')):
    """Map a 'layer' column to 'deep'/'superficial'/None. Shared by Functions 5.4/5.7/5.8/5.9."""
    def _depth(layer):
        if layer in deep_layers:
            return 'deep'
        elif layer in superficial_layers:
            return 'superficial'
        return None
    return layer_series.map(_depth)


def compare_smi_by_depth_group(df, layer_col='layer', group_col='condition', value_col='SMI', filter_col='valid',
                                deep_layers=('L5', 'L6'), superficial_layers=('L2/3', 'L4')):
    """
    Loop compare_smi_across_conditions over the two pooled depth groups
    ('deep' = L5+L6, 'superficial' = L2/3+L4) instead of all four layers.

    Parameters
    ----------
    df : pandas.DataFrame
        One group's table.
    layer_col, group_col, value_col, filter_col : str
    deep_layers, superficial_layers : tuple of str

    Returns
    -------
    results : dict
        {'deep': compare_smi_across_conditions(...) result or None,
         'superficial': ... }.
    summary_df : pandas.DataFrame
        One row per (depth_group, pairwise comparison) that had 2+ categories.
    """
    df = df.copy()
    df['depth_group'] = _assign_depth_group(df[layer_col], deep_layers, superficial_layers)

    results = {}
    summary_rows = []
    for depth_group, layers in (('deep', deep_layers), ('superficial', superficial_layers)):
        print(f"\n--- Depth group: {depth_group} ({'+'.join(layers)}) ---")
        depth_df = df[df['depth_group'] == depth_group]
        result = compare_smi_across_conditions(depth_df, group_col=group_col, value_col=value_col,
                                                filter_col=filter_col)
        results[depth_group] = result
        if result is not None:
            for _, row in result['pairwise'].iterrows():
                summary_rows.append({
                    'depth_group': depth_group, 'cond_a': row['cond_a'], 'cond_b': row['cond_b'],
                    'median_diff': row['median_diff'], 'p_holm': row['p_holm'],
                })

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) > 0:
        print("\n=== Summary: deep vs superficial (each depth group's own Holm correction) ===")
        print(summary_df.to_string(index=False))

    return results, summary_df


# =============================================================================
# Function 5.8 -- plot_smi_by_depth_group
# =============================================================================

def plot_smi_by_depth_group(df, layer_col='layer', group_col='condition', value_col='SMI', filter_col='valid',
                             deep_layers=('L5', 'L6'), superficial_layers=('L2/3', 'L4'), title=''):
    """
    Violin+strip plots, one panel for 'deep' and one for 'superficial'.
    """
    filtered = df[df[filter_col]].copy()
    filtered['depth_group'] = _assign_depth_group(filtered[layer_col], deep_layers, superficial_layers)

    color_by_category = {'baseline': 'tab:blue', 'saline': 'tab:orange', 'dcz': 'tab:green'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, depth_group in zip(axes, ('deep', 'superficial')):
        depth_df = filtered[filtered['depth_group'] == depth_group]
        category_order = [c for c in ['baseline', 'saline', 'dcz'] if c in depth_df[group_col].unique()]
        category_order += [c for c in depth_df[group_col].unique() if c not in category_order]

        data_by_category = [depth_df.loc[depth_df[group_col] == cat, value_col].to_numpy()
                            for cat in category_order]

        if len(data_by_category) == 0 or all(len(d) == 0 for d in data_by_category):
            ax.set_title(f"{depth_group} (no data)")
            ax.axis('off')
            continue

        parts = ax.violinplot(data_by_category, showmedians=True)
        for i, body in enumerate(parts['bodies']):
            body.set_facecolor(color_by_category.get(category_order[i], 'gray'))
            body.set_alpha(0.4)

        rng = np.random.default_rng(0)
        for i, vals in enumerate(data_by_category):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                       color=color_by_category.get(category_order[i], 'gray'), s=12, alpha=0.5)
        ax.set_ylim(-1.3, 1.3)
        ax.set_yticks(np.arange(-1, 1.1, 0.5))
        ax.set_xticks(range(1, len(category_order) + 1))
        ax.set_xticklabels(category_order)
        ax.set_ylabel(value_col)
        ax.set_title(depth_group)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    fig.suptitle(title, fontsize=20, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# Function 5.9 -- summarize_layer_sample_sizes
#
# A diagnostic, not a comparison: tabulates n (valid cells) per layer x
# condition and per depth-group x condition, flagging anything below
# low_n_threshold. Added after noticing saline n=3-6 in several individual
# layers (Active_OL, Stationary_OL especially) while reading Function
# 5.2's output -- makes that caveat visible on every run instead of
# relying on someone noticing it in the printed pairwise tables.
# =============================================================================

def summarize_layer_sample_sizes(df, layer_col='layer', group_col='condition', filter_col='valid',
                                  deep_layers=('L5', 'L6'), superficial_layers=('L2/3', 'L4'),
                                  low_n_threshold=10):
    """
    Tabulate n (valid cells) per layer x condition and per depth_group x
    condition, flagging anything below low_n_threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        One group's table.
    layer_col, group_col, filter_col : str
    deep_layers, superficial_layers : tuple of str
    low_n_threshold : int

    Returns
    -------
    layer_counts : pandas.DataFrame
        One row per layer, one column per condition, n = valid cells.
    depth_counts : pandas.DataFrame
        Same, but for the two pooled depth groups.
    """
    filtered = df[df[filter_col]].copy()
    filtered['depth_group'] = _assign_depth_group(filtered[layer_col], deep_layers, superficial_layers)

    layers_present = [l for l in filtered[layer_col].dropna().unique()]
    layer_order = _layer_order(layers_present)
    layer_counts = filtered.groupby([layer_col, group_col]).size().unstack(fill_value=0).reindex(layer_order)

    depth_counts = filtered.groupby(['depth_group', group_col]).size().unstack(fill_value=0)
    depth_counts = depth_counts.reindex(['deep', 'superficial'])

    print("Sample sizes (valid cells) per layer x condition:")
    print(layer_counts.to_string())
    low_layer = layer_counts[layer_counts.lt(low_n_threshold).any(axis=1)]
    if len(low_layer) > 0:
        print(f"\nWARNING: layer(s) with a condition below n={low_n_threshold} "
              f"-- interpret those specific comparisons cautiously:")
        print(low_layer.to_string())

    print("\nSample sizes (valid cells) per depth group x condition:")
    print(depth_counts.to_string())
    low_depth = depth_counts[depth_counts.lt(low_n_threshold).any(axis=1)]
    if len(low_depth) > 0:
        print(f"\nWARNING: depth group(s) with a condition below n={low_n_threshold}:")
        print(low_depth.to_string())

    return layer_counts, depth_counts


# =============================================================================
# Functions 5.5/5.6 -- drivers
# =============================================================================

def run_layer_analysis_for_group(df, group_name=''):
    """
    Runs Functions 5.9 + 5.2 + 5.3 + 5.7 + 5.8 + 5.4 for one group:
    sample-size diagnostic first, then the 4-layer breakdown, then the
    pooled deep/superficial comparison, then the formal interaction test.
    Retains both figures under 'layer_fig'/'depth_fig' so Function 5.10
    can save them without needing to replot.

    Parameters
    ----------
    df : pandas.DataFrame
        One group's table.
    group_name : str

    Returns
    -------
    result : dict with keys: layer_counts, depth_counts, layer_results,
        layer_summary, layer_fig, depth_results, depth_summary, depth_fig,
        interaction_result.
    """
    print(f"\n{'='*90}\nLayer analysis: {group_name}\n{'='*90}")

    layer_counts, depth_counts = summarize_layer_sample_sizes(df)

    layer_results, layer_summary_df = compare_smi_by_layer(df)
    layer_fig = plot_smi_by_layer(df, title=group_name)
    # plt.show()

    depth_results, depth_summary_df = compare_smi_by_depth_group(df)
    depth_fig = plot_smi_by_depth_group(df, title=f"{group_name} (deep vs superficial)")
    # plt.show()

    interaction_result = test_layer_depth_interaction(df)

    return {
        'layer_counts': layer_counts,
        'depth_counts': depth_counts,
        'layer_results': layer_results,
        'layer_summary': layer_summary_df,
        'layer_fig': layer_fig,
        'depth_results': depth_results,
        'depth_summary': depth_summary_df,
        'depth_fig': depth_fig,
        'interaction_result': interaction_result,
    }


def run_layer_analysis_all_groups(all_group_dfs, group_col='condition', filter_col='valid'):
    """
    Loops run_layer_analysis_for_group over every group, skipping
    single-condition groups (no condition contrast to test against depth
    -- e.g. the 'baseline' group).

    Parameters
    ----------
    all_group_dfs : dict
        {group_name: df}, from load_all_group_dfs_from_phase4.
    group_col, filter_col : str

    Returns
    -------
    results : dict
        {group_name: run_layer_analysis_for_group(...) result}.
    """
    results = {}
    for group_name, df in all_group_dfs.items():
        conditions_present = df.loc[df[filter_col], group_col].unique()
        if len(conditions_present) < 2:
            print(f"\n{'='*90}\n{group_name}: only {len(conditions_present)} condition(s) present "
                  f"({list(conditions_present)}) -- skipping layer analysis for this group.\n{'='*90}")
            continue
        results[group_name] = run_layer_analysis_for_group(df, group_name=group_name)

    return results


# =============================================================================
# Function 5.10 -- save everything Phase 5 generates
#
# Same reasoning as Phase 4's Function 4.8: nothing here was persisted to
# disk until this function was added -- the sample-size tables, layer/depth
# stats, both figures per group, and the interaction regression only ever
# lived in the kernel's memory.
# =============================================================================

def save_dataframe_csv(df, output_dir, filename, index=False):
    """
    Save a DataFrame to {output_dir}/{filename}, creating output_dir if
    needed. index=True for tables whose index is meaningful (e.g. layer
    names), False for tables with a plain range index.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    df.to_csv(save_path, index=index)
    print(f"Saved -> {save_path}")
    return save_path


def save_figure_png(fig, output_dir, filename, dpi=150):
    """
    Save a matplotlib figure to {output_dir}/{filename}, creating
    output_dir if needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved -> {save_path}")
    return save_path


def _json_safe(obj):
    """Recursively convert numpy scalar types to native Python for json.dump."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json(data, output_dir, filename):
    """
    Save a JSON-serializable dict to {output_dir}/{filename}, creating
    output_dir if needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    with open(save_path, 'w') as f:
        json.dump(_json_safe(data), f, indent=2)
    print(f"Saved -> {save_path}")
    return save_path


def _extract_omnibus_summary(results_by_category):
    """
    Pull {category: {'omnibus_stat', 'omnibus_p', 'group_medians', 'group_n'}}
    out of a {category: compare_smi_across_conditions result or None} dict,
    for JSON saving (skips categories where result was None).
    """
    summary = {}
    for cat, result in results_by_category.items():
        if result is None:
            continue
        summary[cat] = {
            'omnibus_stat': result['omnibus_stat'],
            'omnibus_p': result['omnibus_p'],
            'group_medians': result['group_medians'],
            'group_n': result['group_n'],
        }
    return summary


def save_layer_analysis_group_outputs(output_dir, group_name, result):
    """
    Save one group's layer-analysis outputs: sample-size tables, per-layer
    and per-depth-group summary stats (+ omnibus JSON), both figures, and
    the interaction regression's text summary.

    Saved: {group}_layer_sample_sizes.csv, {group}_depth_sample_sizes.csv,
    {group}_layer_pairwise_stats.csv, {group}_depth_pairwise_stats.csv,
    {group}_layer_omnibus_stats.json, {group}_depth_omnibus_stats.json,
    {group}_layer_comparison_plot.png, {group}_depth_comparison_plot.png,
    {group}_interaction_regression.txt.

    Parameters
    ----------
    output_dir : str
    group_name : str
    result : dict
        From run_layer_analysis_for_group (carries 'layer_fig'/'depth_fig').

    Returns
    -------
    saved_paths : dict
    """
    saved_paths = {
        'layer_sample_sizes': save_dataframe_csv(
            result['layer_counts'], output_dir, f"{group_name}_layer_sample_sizes.csv", index=True),
        'depth_sample_sizes': save_dataframe_csv(
            result['depth_counts'], output_dir, f"{group_name}_depth_sample_sizes.csv", index=True),
    }

    if len(result['layer_summary']) > 0:
        saved_paths['layer_pairwise'] = save_dataframe_csv(
            result['layer_summary'], output_dir, f"{group_name}_layer_pairwise_stats.csv")
    if len(result['depth_summary']) > 0:
        saved_paths['depth_pairwise'] = save_dataframe_csv(
            result['depth_summary'], output_dir, f"{group_name}_depth_pairwise_stats.csv")

    saved_paths['layer_omnibus'] = save_json(
        _extract_omnibus_summary(result['layer_results']), output_dir, f"{group_name}_layer_omnibus_stats.json")
    saved_paths['depth_omnibus'] = save_json(
        _extract_omnibus_summary(result['depth_results']), output_dir, f"{group_name}_depth_omnibus_stats.json")

    layer_fig = result.get('layer_fig')
    if layer_fig is not None:
        saved_paths['layer_plot'] = save_figure_png(layer_fig, output_dir, f"{group_name}_layer_comparison_plot.png")

    depth_fig = result.get('depth_fig')
    if depth_fig is not None:
        saved_paths['depth_plot'] = save_figure_png(depth_fig, output_dir, f"{group_name}_depth_comparison_plot.png")

    interaction_result = result.get('interaction_result')
    if interaction_result is not None:
        os.makedirs(output_dir, exist_ok=True)
        txt_path = os.path.join(output_dir, f"{group_name}_interaction_regression.txt")
        with open(txt_path, 'w') as f:
            f.write(str(interaction_result.summary()))
        print(f"Saved -> {txt_path}")
        saved_paths['interaction_regression'] = txt_path

    return saved_paths


def save_all_layer_analysis_outputs(output_dir, layer_analysis_results):
    """
    Loop save_layer_analysis_group_outputs over every group.

    Parameters
    ----------
    output_dir : str
    layer_analysis_results : dict
        {group_name: run_layer_analysis_for_group(...) result}.

    Returns
    -------
    saved_paths_by_group : dict
    """
    saved_paths_by_group = {}
    for group_name, result in layer_analysis_results.items():
        saved_paths_by_group[group_name] = save_layer_analysis_group_outputs(output_dir, group_name, result)
    print(f"\nSaved layer-analysis outputs for {len(saved_paths_by_group)} group(s) to {output_dir}")
    return saved_paths_by_group


# =============================================================================
# Function 5.11 -- compute_per_group_layer_smi_summary
#
# Function 5.12 -- test_paired_smi_significance_by_layer
#
# Function 5.13 -- test_layer_heterogeneity_in_smi_diff
#
# The corrected version of "is there a significant saline-vs-dcz SMI
# difference in each layer, and does the SIZE of that difference differ
# significantly between layers." Functions 5.1/5.2/5.4 pool CELLS within
# a layer as independent samples, but for any one comparison group
# there's only one saline session and one paired dcz session -- cells
# within it aren't independent replicates of that condition. The correct
# independent unit is the comparison GROUP itself (5 of them: DCZ1/2/3,
# Active_OL, Stationary_OL), same fix as Phase 6's paired t-test approach
# (test_paired_significance_by_layer there). A paired t-test uses
# magnitude and consistency, not just sign, so it isn't capped the way a
# Wilcoxon signed-rank is at small n.
# =============================================================================

def compute_per_group_layer_smi_summary(all_group_dfs, layer_col='layer', group_col='condition',
                                         value_col='SMI', filter_col='valid'):
    """
    Per comparison group x layer: median SMI for saline and dcz cells
    (valid-filtered), plus n. This is the per-group-per-layer summary
    statistic the paired tests below need -- each group is ONE
    independent saline/dcz pair. See markdown/section comment above.

    Also includes 'within_group_mannwhitney_p' -- Mann-Whitney U on that
    ONE recording's saline vs dcz cells (same test as Function 5.1, just
    restricted to the saline/dcz pair). This is a per-recording
    breakdown, not an independent significance test: cells within a
    single session are pseudo-replicated the same way Functions 5.1/5.2
    always were, so this column shouldn't be read as its own reliable
    p-value -- it's here to show which specific recording is driving (or
    not driving) the aggregate pattern from the paired t-test
    (Function 5.12), the way median_diff already does, not to replace it.

    Parameters
    ----------
    all_group_dfs : dict
        {group_name: df}, from load_all_group_dfs_from_phase4.
    layer_col, group_col, value_col, filter_col : str

    Returns
    -------
    summary_df : pandas.DataFrame
        One row per (group, layer) where both saline and dcz are present:
        group, layer, saline_median, dcz_median, diff, n_saline, n_dcz,
        within_group_mannwhitney_p.
    """
    rows = []
    for group_name, df in all_group_dfs.items():
        filtered = df[df[filter_col]]
        conditions_present = filtered[group_col].unique()
        if 'saline' not in conditions_present or 'dcz' not in conditions_present:
            continue

        layers_present = [l for l in filtered[layer_col].dropna().unique()]
        for layer in _layer_order(layers_present):
            layer_df = filtered[filtered[layer_col] == layer]
            saline_vals = layer_df.loc[layer_df[group_col] == 'saline', value_col].to_numpy()
            dcz_vals = layer_df.loc[layer_df[group_col] == 'dcz', value_col].to_numpy()
            if len(saline_vals) == 0 or len(dcz_vals) == 0:
                continue
            saline_median = float(np.median(saline_vals))
            dcz_median = float(np.median(dcz_vals))

            if len(saline_vals) >= 1 and len(dcz_vals) >= 1:
                try:
                    _, p_within_group = mannwhitneyu(saline_vals, dcz_vals, alternative='two-sided')
                except ValueError:
                    p_within_group = np.nan
            else:
                p_within_group = np.nan

            rows.append({
                'group': group_name, 'layer': layer,
                'saline_median': saline_median, 'dcz_median': dcz_median,
                'diff': saline_median - dcz_median,
                'n_saline': len(saline_vals), 'n_dcz': len(dcz_vals),
                'within_group_mannwhitney_p': p_within_group,
            })

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    print("\nNOTE: 'within_group_mannwhitney_p' treats cells within one recording as independent")
    print("samples (pseudo-replicated, same issue as Functions 5.1/5.2) -- descriptive, not a")
    print("reliable standalone significance test the way Function 5.12's paired t-test is.")
    return summary_df


def test_paired_smi_significance_by_layer(summary_df):
    """
    Paired t-test (+ Wilcoxon, for comparison) on median SMI, saline vs
    dcz, per layer, across the independent comparison groups. See
    section comment above.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        From compute_per_group_layer_smi_summary.

    Returns
    -------
    result_df : pandas.DataFrame
        One row per layer: n_pairs, mean_diff, std_diff, n_same_direction,
        paired_t_p, wilcoxon_p.
    """
    rows = []
    for layer in _layer_order(summary_df['layer'].unique()):
        layer_rows = summary_df[summary_df['layer'] == layer]
        saline_vals = layer_rows['saline_median'].to_numpy()
        dcz_vals = layer_rows['dcz_median'].to_numpy()
        n_pairs = len(saline_vals)

        if n_pairs < 2:
            print(f"{layer}: only {n_pairs} pair(s) -- skipping (need >=2 for a paired test).")
            continue

        diffs = saline_vals - dcz_vals
        t_stat, p_ttest = ttest_rel(saline_vals, dcz_vals)
        if np.all(diffs == diffs[0]):
            w_stat, p_wilcoxon = np.nan, np.nan
        else:
            w_stat, p_wilcoxon = wilcoxon(saline_vals, dcz_vals)

        rows.append({
            'layer': layer, 'n_pairs': n_pairs,
            'mean_diff': diffs.mean(), 'std_diff': diffs.std(ddof=1) if n_pairs > 1 else np.nan,
            'n_same_direction': int((diffs > 0).sum()),
            'paired_t_stat': t_stat, 'paired_t_p': p_ttest, 'wilcoxon_p': p_wilcoxon,
        })

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))
    return result_df


def test_layer_heterogeneity_in_smi_diff(summary_df, diff_col='diff'):
    """
    Does the SIZE of the saline-dcz SMI difference vary significantly
    across layers? Repeated-measures ANOVA (subject = group/pair,
    within-factor = layer) as the omnibus test, then Holm-corrected
    pairwise paired t-tests between layers for the breakdown -- same
    omnibus-then-pairwise structure as every other test in this phase,
    using the design appropriate for n=independent-pairs data. This
    specific question (does the effect differ significantly BETWEEN
    layers, not just whether each layer's own saline-vs-dcz test is
    significant) was never actually tested in this phase before -- 5.4's
    interaction test only compared two pooled buckets (deep/superficial),
    not all four layers, and was cell-level (pseudo-replicated) besides.

    Only uses groups that have all layers present (AnovaRM needs balanced
    data) -- prints a warning listing any group dropped for this reason.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        From compute_per_group_layer_smi_summary.
    diff_col : str

    Returns
    -------
    aov_result : statsmodels AnovaRM results object, or None if fewer
        than 2 complete groups.
    pairwise_df : pandas.DataFrame
        One row per layer pair: mean_diff_a, mean_diff_b, t_stat, p_raw, p_holm.
    """
    pivot = summary_df.pivot(index='group', columns='layer', values=diff_col)
    complete = pivot.dropna()
    dropped = set(pivot.index) - set(complete.index)
    if dropped:
        print(f"Dropped {len(dropped)} group(s) missing one or more layers "
              f"(AnovaRM needs balanced data): {sorted(dropped)}")

    if len(complete) < 2:
        print(f"Only {len(complete)} complete group(s) -- skipping (need >=2).")
        return None, pd.DataFrame()

    rm_df = complete.reset_index().melt(id_vars='group', var_name='layer', value_name=diff_col)

    aov_result = AnovaRM(rm_df, depvar=diff_col, subject='group', within=['layer']).fit()
    print(f"\n=== Repeated-measures ANOVA: does the saline-dcz {diff_col} vary by layer? ===")
    print(aov_result)

    layers = list(complete.columns)
    pairwise_rows = []
    for layer_a, layer_b in combinations(layers, 2):
        t_stat, p_raw = ttest_rel(complete[layer_a], complete[layer_b])
        pairwise_rows.append({
            'layer_a': layer_a, 'layer_b': layer_b,
            'mean_diff_a': complete[layer_a].mean(), 'mean_diff_b': complete[layer_b].mean(),
            't_stat': t_stat, 'p_raw': p_raw,
        })
    pairwise_df = pd.DataFrame(pairwise_rows)
    if len(pairwise_df) > 0:
        _, p_holm, _, _ = multipletests(pairwise_df['p_raw'], method='holm')
        pairwise_df['p_holm'] = p_holm

    print(f"\nPairwise layer-vs-layer comparison of the saline-dcz {diff_col} (Holm-corrected):")
    print(pairwise_df.to_string(index=False))

    return aov_result, pairwise_df


# Fixed categorical color per comparison group -- never cycled/re-painted.
# Same hex values as Phase 6's GROUP_COLORS, deliberately, so the same
# group (e.g. "DCZ2") reads as the same color across both phases' figures.
GROUP_COLORS = {
    'DCZ1': '#1b9e77',
    'DCZ2': '#d95f02',
    'DCZ3': '#7570b3',
    'Active_OL': '#e7298a',
    'Stationary_OL': '#66a61e',
}


def plot_paired_smi_by_layer(summary_df, title=''):
    """
    Small multiples (one panel per layer): one line per comparison group,
    saline -> dcz median SMI. Same slope-plot design as Phase 6's
    plot_landmark_last_slope_by_layer, adapted for this phase's
    per-group-per-layer SMI summary.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        From compute_per_group_layer_smi_summary.
    title : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    layer_order = _layer_order(summary_df['layer'].unique())
    groups = sorted(summary_df['group'].unique())

    fig, axes = plt.subplots(1, len(layer_order), figsize=(5 * len(layer_order), 7.5), sharey=True)
    axes = np.atleast_1d(axes)
    x_positions = {'saline': 0, 'dcz': 1}

    for ax, layer in zip(axes, layer_order):
        layer_rows = summary_df[summary_df['layer'] == layer]
        for _, row in layer_rows.iterrows():
            color = GROUP_COLORS.get(row['group'], 'gray')
            ax.plot([x_positions['saline'], x_positions['dcz']],
                    [row['saline_median'], row['dcz_median']],
                    color=color, marker='o', markersize=8, linewidth=2,
                    solid_capstyle='round', zorder=3)

        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['saline', 'dcz'])
        ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_title(layer)

    axes[0].set_ylabel('Median SMI')

    handles = [plt.Line2D([0], [0], color=GROUP_COLORS.get(g, 'gray'), marker='o', linewidth=2, label=g)
              for g in groups]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=11,
              bbox_to_anchor=(0.5, -0.06), frameon=False)

    fig.suptitle(title if title else 'Median SMI, saline -> dcz, by layer (paired by comparison group)')
    plt.tight_layout()
    return fig


def run_paired_layer_smi_analysis(all_group_dfs):
    """
    Runs the corrected (n=independent pairs, not n=pooled cells) version
    of Phase 5's layer-specific SMI question: 5.11 -> 5.12 -> 5.13 -> 5.14.

    Parameters
    ----------
    all_group_dfs : dict
        {group_name: df}, from load_all_group_dfs_from_phase4.

    Returns
    -------
    summary_df : pandas.DataFrame
    layer_significance_df : pandas.DataFrame
    aov_result : statsmodels AnovaRM results object or None
    layer_heterogeneity_df : pandas.DataFrame
    fig : matplotlib.figure.Figure
    """
    print(f"\n{'='*90}\nCorrected paired-significance layer SMI analysis\n{'='*90}")

    summary_df = compute_per_group_layer_smi_summary(all_group_dfs)
    layer_significance_df = test_paired_smi_significance_by_layer(summary_df)
    aov_result, layer_heterogeneity_df = test_layer_heterogeneity_in_smi_diff(summary_df)
    fig = plot_paired_smi_by_layer(summary_df)
    # plt.show()

    return summary_df, layer_significance_df, aov_result, layer_heterogeneity_df, fig


def save_paired_layer_smi_outputs(output_dir, summary_df, layer_significance_df,
                                  aov_result, layer_heterogeneity_df, fig):
    """
    Saves Functions 5.11-5.14's outputs: paired_layer_smi_summary.csv,
    paired_layer_smi_significance.csv, layer_heterogeneity_pairwise.csv,
    layer_heterogeneity_anova.txt, paired_layer_smi_plot.png.

    Parameters
    ----------
    output_dir : str
    summary_df, layer_significance_df, layer_heterogeneity_df : pandas.DataFrame
    aov_result : statsmodels AnovaRM results object or None
    fig : matplotlib.figure.Figure or None

    Returns
    -------
    saved_paths : dict
    """
    saved_paths = {
        'summary': save_dataframe_csv(summary_df, output_dir, 'paired_layer_smi_summary.csv'),
        'significance': save_dataframe_csv(layer_significance_df, output_dir, 'paired_layer_smi_significance.csv'),
    }
    if len(layer_heterogeneity_df) > 0:
        saved_paths['heterogeneity'] = save_dataframe_csv(
            layer_heterogeneity_df, output_dir, 'layer_heterogeneity_pairwise.csv')
    if aov_result is not None:
        os.makedirs(output_dir, exist_ok=True)
        txt_path = os.path.join(output_dir, 'layer_heterogeneity_anova.txt')
        with open(txt_path, 'w') as f:
            f.write(str(aov_result))
        print(f"Saved -> {txt_path}")
        saved_paths['heterogeneity_anova'] = txt_path
    if fig is not None:
        saved_paths['plot'] = save_figure_png(fig, output_dir, 'paired_layer_smi_plot.png')

    print(f"\nSaved paired layer-SMI analysis outputs to {output_dir}")
    return saved_paths


# =============================================================================
# Driver
# =============================================================================

if __name__ == '__main__':
    # Point this at whichever animal you're processing.
    ANIMAL_DIR = r"D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD"
    PHASE4_OUTPUT_DIR = os.path.join(ANIMAL_DIR, 'Phase4_SessionComparison_Results')
    PHASE5_OUTPUT_DIR = os.path.join(ANIMAL_DIR, 'Phase5_LayerSpecific_Results')

    all_group_dfs = load_all_group_dfs_from_phase4(PHASE4_OUTPUT_DIR)
    layer_analysis_results = run_layer_analysis_all_groups(all_group_dfs)
    saved_paths = save_all_layer_analysis_outputs(PHASE5_OUTPUT_DIR, layer_analysis_results)

    # --- Corrected, properly-powered version of the layer-specific SMI
    #     question (n=independent comparison groups, not n=pooled cells) ---
    (paired_layer_smi_summary_df, paired_layer_smi_significance_df,
     layer_heterogeneity_aov, layer_heterogeneity_pairwise_df,
     paired_layer_smi_fig) = run_paired_layer_smi_analysis(all_group_dfs)
    paired_smi_saved_paths = save_paired_layer_smi_outputs(
        PHASE5_OUTPUT_DIR, paired_layer_smi_summary_df, paired_layer_smi_significance_df,
        layer_heterogeneity_aov, layer_heterogeneity_pairwise_df, paired_layer_smi_fig)
