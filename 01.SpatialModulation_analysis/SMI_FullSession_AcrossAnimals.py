"""
SMICalculation_AnalyzeAcrossAnimals.py

Cross-animal SMI comparison: data loading, statistical analyses, and visualization.

Analyses (pooled across animals):
1. Day 1 layer differences - do deeper layers start with higher SMI?
2. Temporal progression per layer - does SMI increase over days?
3. Layer development rate comparisons - do layers develop at different rates?
4. Early (Days 1-2) vs Late (Days 6-7) comparison
5. Gap closure - does the Deep - Superficial gap narrow over time?
6. Individual animal trajectory tracking

Statistics:
- Kruskal-Wallis + Mann-Whitney pairwise with FDR correction
- Cliff's Delta effect sizes (non-parametric)
- Bootstrap confidence intervals
- Permutation tests (slope, slope comparison, group differences)

Input: SMI result files (*_smi_results.h5) from all animals
Output: Statistical summaries, 3x3 publication figure, CSV tables

JSY, 2025
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
import seaborn as sns
from scipy import stats
import h5py
from glob import glob
from collections import defaultdict
import pandas as pd
from itertools import combinations


# ============================================================================
# CONFIGURATION
# ============================================================================

LAYER_ORDER = ['L2/3', 'L4', 'L5', 'L6']
LAYER_COLORS = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}


# ============================================================================
# STATISTICAL UTILITY FUNCTIONS
# ============================================================================

def cliffs_delta(x, y):
    """
    Cliff's Delta non-parametric effect size.
    |delta| < 0.147 = negligible, < 0.33 = small, < 0.474 = medium, >= 0.474 = large
    """
    if len(x) == 0 or len(y) == 0:
        return np.nan, "undefined"

    n_x, n_y = len(x), len(y)
    greater = np.sum([np.sum(x_i > y) for x_i in x])
    less = np.sum([np.sum(x_i < y) for x_i in x])
    delta = (greater - less) / (n_x * n_y)

    abs_delta = abs(delta)
    if abs_delta < 0.147:
        magnitude = "negligible"
    elif abs_delta < 0.33:
        magnitude = "small"
    elif abs_delta < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"

    return delta, magnitude


def bootstrap_ci(data, statistic=np.median, n_bootstrap=10000, ci=95, paired=False):
    """Bootstrap confidence interval for a statistic."""
    if len(data) == 0:
        return (np.nan, np.nan)

    data = np.array(data)
    bootstrap_stats = []

    if paired and data.ndim == 2:
        n_samples = data.shape[0]
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_stats.append(statistic(data[indices]))
    else:
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic(bootstrap_sample))

    bootstrap_stats = np.array(bootstrap_stats)
    lower = np.percentile(bootstrap_stats, (100 - ci) / 2)
    upper = np.percentile(bootstrap_stats, 100 - (100 - ci) / 2)
    return (lower, upper)


def permutation_test(group1, group2, n_permutations=10000, statistic='median_diff'):
    """Permutation test for difference between two groups (two-tailed)."""
    if len(group1) == 0 or len(group2) == 0:
        return np.nan, np.nan

    if statistic == 'median_diff':
        observed_stat = np.median(group1) - np.median(group2)
    else:
        observed_stat = np.mean(group1) - np.mean(group2)

    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    perm_stats = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        if statistic == 'median_diff':
            perm_stats.append(np.median(combined[:n1]) - np.median(combined[n1:]))
        else:
            perm_stats.append(np.mean(combined[:n1]) - np.mean(combined[n1:]))

    p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
    return p_value, observed_stat


def fdr_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction."""
    p_values = np.array(p_values)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    reject = np.zeros(n, dtype=bool)
    for i, p in enumerate(sorted_p):
        if p <= (i + 1) / n * alpha:
            reject[sorted_indices[i]] = True

    p_adjusted = np.minimum(sorted_p * n / np.arange(1, n + 1), 1.0)
    p_adjusted = np.minimum.accumulate(p_adjusted[::-1])[::-1]
    original_order = np.argsort(sorted_indices)
    p_adjusted = p_adjusted[original_order]

    return reject, p_adjusted


def permutation_slope_test(days, medians, n_permutations=10000):
    """Permutation test for slope significance (H0: slope = 0)."""
    if len(days) < 2:
        return np.nan, np.nan

    slope, _, _, _, _ = stats.linregress(days, medians)
    perm_slopes = []
    for _ in range(n_permutations):
        perm_medians = np.random.permutation(medians)
        perm_slope, _, _, _, _ = stats.linregress(days, perm_medians)
        perm_slopes.append(perm_slope)

    p_value = np.mean(np.abs(perm_slopes) >= np.abs(slope))
    return p_value, slope


def permutation_compare_slopes(days1, medians1, days2, medians2, n_permutations=10000):
    """Permutation test comparing slopes between two groups (H0: slope1 = slope2)."""
    if len(days1) < 2 or len(days2) < 2:
        return np.nan, np.nan

    slope1, _, _, _, _ = stats.linregress(days1, medians1)
    slope2, _, _, _, _ = stats.linregress(days2, medians2)
    observed_diff = slope1 - slope2

    all_days = np.concatenate([days1, days2])
    all_medians = np.concatenate([medians1, medians2])
    all_labels = np.concatenate([np.zeros(len(days1)), np.ones(len(days2))])

    perm_diffs = []
    for _ in range(n_permutations):
        perm_labels = np.random.permutation(all_labels)
        g1 = perm_labels == 0
        g2 = perm_labels == 1
        if np.sum(g1) >= 2 and np.sum(g2) >= 2:
            s1, _, _, _, _ = stats.linregress(all_days[g1], all_medians[g1])
            s2, _, _, _, _ = stats.linregress(all_days[g2], all_medians[g2])
            perm_diffs.append(s1 - s2)

    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    return p_value, observed_diff


# ============================================================================
# DATA LOADING
# ============================================================================

def extract_info(filepath):
    """Extract animal ID and day from filepath."""
    match_day = re.search(r'Day(\d+)', filepath, re.IGNORECASE)
    match_animal = re.search(r'(JSY\d+)', filepath)
    day = int(match_day.group(1)) if match_day else None
    animal = match_animal.group(1) if match_animal else None
    return animal, day


def load_all_animals_smi(parent_dir, smi_pattern="*_smi_results.h5"):
    """Load SMI data from all animals under parent_dir."""
    print(f"\n{'='*70}")
    print("LOADING ALL ANIMALS SMI DATA")
    print(f"{'='*70}")
    print(f"Parent directory: {parent_dir}\n")

    smi_files = glob(os.path.join(parent_dir, "**", smi_pattern), recursive=True)
    print(f"Found {len(smi_files)} SMI result files\n")

    all_data = defaultdict(lambda: defaultdict(dict))

    for smi_path in sorted(smi_files):
        animal_id, day = extract_info(smi_path)
        if animal_id is None or day is None:
            continue

        print(f"  Loading {animal_id} Day {day}")

        try:
            with h5py.File(smi_path, 'r') as f:
                session_data = {'day': day, 'layers': {}}

                if 'global_smi' in f:
                    g = f['global_smi']
                    session_data['global_median'] = g.attrs.get('median_smi', np.nan)
                    session_data['global_n'] = g.attrs.get('n_valid_cells', 0)

                if 'layer_smi' in f:
                    for layer_key in f['layer_smi'].keys():
                        lg = f['layer_smi'][layer_key]
                        layer_name = lg.attrs.get('original_name', layer_key.replace('_', '/'))
                        smi_vals = lg['SMI'][:] if 'SMI' in lg else np.array([])

                        prop_mod = np.mean(smi_vals > 0.1) if len(smi_vals) > 0 else np.nan

                        session_data['layers'][layer_name] = {
                            'n_valid': lg.attrs.get('n_cells_valid', len(smi_vals)),
                            'median_smi': lg.attrs.get('median_smi', np.nan),
                            'mean_smi': lg.attrs.get('mean_smi', np.nan),
                            'mad_smi': lg.attrs.get('mad_smi', np.nan),
                            'SMI': smi_vals,
                            'prop_modulated': prop_mod,
                            'preferred_positions': lg['preferred_positions'][:] if 'preferred_positions' in lg else np.array([]),
                            'cell_indices': lg['cell_indices'][:].astype(int) if 'cell_indices' in lg else np.array([], dtype=int),
                            'reliable_valid_cells': lg['reliable_valid_cells'][:].astype(int) if 'reliable_valid_cells' in lg else np.array([], dtype=int),
                        }

                if 'cell_info' in f:
                    session_data['med_coords'] = f['cell_info']['med_coords'][:] if 'med_coords' in f['cell_info'] else None
                else:
                    session_data['med_coords'] = None

                all_data[animal_id][day] = session_data

        except Exception as e:
            print(f"  ERROR: {e}")

    animals = list(all_data.keys())
    print(f"\nLoaded {len(animals)} animals:")
    for animal in sorted(animals):
        days = sorted(all_data[animal].keys())
        print(f"  {animal}: Days {days}")

    return dict(all_data)


# ============================================================================
# ANALYSIS 1: DAY 1 LAYER DIFFERENCES
# ============================================================================

def analyze_day1_layer_differences_pooled(all_data):
    """
    Q1: On Day 1, do deeper layers have higher SMI? (Pooled across animals)

    Tests: Kruskal-Wallis, Mann-Whitney pairwise + FDR, Cliff's Delta,
           permutation test for Superficial vs Deep, bootstrap CI.
    """
    print(f"\n{'='*70}")
    print("ANALYSIS 1: DAY 1 LAYER DIFFERENCES (POOLED)")
    print(f"{'='*70}")

    pooled_smi = {layer: [] for layer in LAYER_ORDER}
    animal_medians = {layer: [] for layer in LAYER_ORDER}

    for _, animal_data in all_data.items():
        day1 = min(animal_data.keys())
        if day1 > 2:
            continue
        session = animal_data[day1]
        for layer in LAYER_ORDER:
            if layer in session['layers'] and len(session['layers'][layer]['SMI']) > 0:
                smi_vals = session['layers'][layer]['SMI']
                pooled_smi[layer].extend(smi_vals)
                animal_medians[layer].append(session['layers'][layer]['median_smi'])

    print("\nPooled SMI values (Day 1):")
    for layer in LAYER_ORDER:
        if len(pooled_smi[layer]) > 0:
            med = np.median(pooled_smi[layer])
            mad = stats.median_abs_deviation(pooled_smi[layer])
            n = len(pooled_smi[layer])
            n_animals = len(animal_medians[layer])
            print(f"  {layer}: n={n} cells from {n_animals} animals, median={med:.3f}±{mad:.3f}")

    valid_layers = [lyr for lyr in LAYER_ORDER if len(pooled_smi[lyr]) > 0]
    results = {'pooled_smi': pooled_smi, 'animal_medians': animal_medians}

    if len(valid_layers) >= 2:
        # Kruskal-Wallis
        groups = [pooled_smi[lyr] for lyr in valid_layers]
        h_stat, kw_p = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis (pooled cells): H={h_stat:.3f}, p={kw_p:.6f}")
        results['kruskal_wallis'] = {'H': h_stat, 'p': kw_p}

        # Pairwise Mann-Whitney + FDR + Cliff's Delta
        print("\nPairwise comparisons (Mann-Whitney U + FDR correction):")
        pairwise = {}
        p_values_list = []
        comparison_names = []

        for l1, l2 in combinations(valid_layers, 2):
            u_stat, p_val = stats.mannwhitneyu(pooled_smi[l1], pooled_smi[l2],
                                               alternative='two-sided')
            delta, magnitude = cliffs_delta(pooled_smi[l1], pooled_smi[l2])
            comparison_names.append(f'{l1}_vs_{l2}')
            p_values_list.append(p_val)
            pairwise[f'{l1}_vs_{l2}'] = {'U': u_stat, 'p': p_val,
                                          'cliffs_delta': delta, 'effect_magnitude': magnitude}

        reject_fdr, p_adjusted = fdr_correction(p_values_list, alpha=0.05)
        for i, comp_name in enumerate(comparison_names):
            pairwise[comp_name]['p_fdr'] = p_adjusted[i]
            pairwise[comp_name]['significant_fdr'] = reject_fdr[i]
            p_raw = pairwise[comp_name]['p']
            p_fdr = p_adjusted[i]
            delta = pairwise[comp_name]['cliffs_delta']
            mag = pairwise[comp_name]['effect_magnitude']
            sig_raw = '***' if p_raw < 0.001 else '**' if p_raw < 0.01 else '*' if p_raw < 0.05 else ''
            sig_fdr = '***' if p_fdr < 0.001 else '**' if p_fdr < 0.01 else '*' if p_fdr < 0.05 else ''
            print(f"  {comp_name}: p={p_raw:.6f}{sig_raw}, p_FDR={p_fdr:.6f}{sig_fdr}, "
                  f"delta={delta:.3f} ({mag})")

        results['pairwise'] = pairwise

        # Superficial vs Deep (permutation + bootstrap CI)
        if 'L2/3' in valid_layers and ('L5' in valid_layers or 'L6' in valid_layers):
            sup_smi = pooled_smi['L2/3']
            deep_smi = []
            for lyr in ['L5', 'L6']:
                if lyr in valid_layers:
                    deep_smi.extend(pooled_smi[lyr])

            u_stat, p_mw = stats.mannwhitneyu(sup_smi, deep_smi, alternative='two-sided')
            p_perm, obs_diff = permutation_test(deep_smi, sup_smi, n_permutations=10000)
            delta, magnitude = cliffs_delta(deep_smi, sup_smi)
            ci_lower, ci_upper = bootstrap_ci(np.array(deep_smi) - np.median(sup_smi),
                                              statistic=np.median, n_bootstrap=10000)

            print(f"\n{'='*70}")
            print("SUPERFICIAL (L2/3) vs DEEP (L5+L6)")
            print(f"{'='*70}")
            print(f"  L2/3 median: {np.median(sup_smi):.3f}")
            print(f"  Deep median: {np.median(deep_smi):.3f}")
            print(f"  Difference: {obs_diff:.3f}")
            print(f"  Mann-Whitney U: p={p_mw:.6f}")
            print(f"  Permutation test: p={p_perm:.6f}")
            print(f"  Cliff's Delta: delta={delta:.3f} ({magnitude})")
            print(f"  95% CI (median diff): [{ci_lower:.3f}, {ci_upper:.3f}]")

            results['superficial_vs_deep'] = {
                'p_mannwhitney': p_mw,
                'p_permutation': p_perm,
                'observed_diff': obs_diff,
                'cliffs_delta': delta,
                'effect_magnitude': magnitude,
                'ci_95': (ci_lower, ci_upper),
                'sup_median': np.median(sup_smi),
                'deep_median': np.median(deep_smi)
            }

    return results


# ============================================================================
# ANALYSIS 2: TEMPORAL PROGRESSION
# ============================================================================

def analyze_temporal_progression_pooled(all_data):
    """
    Q2: Does SMI increase over days in each layer? (Pooled)

    Tests: Spearman correlation + bootstrap CI, linear regression,
           permutation test for slope, first vs last day Mann-Whitney + Cliff's Delta.
    """
    print(f"\n{'='*70}")
    print("ANALYSIS 2: TEMPORAL PROGRESSION (POOLED)")
    print(f"{'='*70}")

    day_layer_data = defaultdict(lambda: {lyr: [] for lyr in LAYER_ORDER})
    all_days = set()

    for animal_id, animal_data in all_data.items():
        for day, session in animal_data.items():
            all_days.add(day)
            for layer in LAYER_ORDER:
                if layer in session['layers'] and len(session['layers'][layer]['SMI']) > 0:
                    day_layer_data[day][layer].extend(session['layers'][layer]['SMI'])

    days_sorted = sorted(all_days)
    results = {'by_layer': {}, 'days': days_sorted}

    for layer in LAYER_ORDER:
        print(f"\n{'='*70}")
        print(f"{layer}")
        print(f"{'='*70}")

        days_with_data = []
        medians = []
        n_cells = []

        for day in days_sorted:
            smi_vals = day_layer_data[day][layer]
            if len(smi_vals) > 0:
                days_with_data.append(day)
                medians.append(np.median(smi_vals))
                n_cells.append(len(smi_vals))
                print(f"  Day {day}: median={np.median(smi_vals):.3f} (n={len(smi_vals)})")

        if len(days_with_data) < 2:
            print("  Insufficient data for trend analysis")
            continue

        days_arr = np.array(days_with_data)
        medians_arr = np.array(medians)

        # Spearman + bootstrap CI
        rho, p_spearman = stats.spearmanr(days_arr, medians_arr)

        def spearman_stat(data):
            return stats.spearmanr(data[:, 0], data[:, 1])[0]

        data_combined = np.column_stack([days_arr, medians_arr])
        rho_ci = bootstrap_ci(data_combined, statistic=spearman_stat,
                              n_bootstrap=5000, paired=True)

        print(f"\nSpearman: rho={rho:.3f}, p={p_spearman:.4f}, 95% CI: [{rho_ci[0]:.3f}, {rho_ci[1]:.3f}]")

        # Linear regression + permutation slope test
        slope, _, r_val, p_linreg, _ = stats.linregress(days_arr, medians_arr)
        p_perm_slope, _ = permutation_slope_test(days_arr, medians_arr, n_permutations=10000)

        print(f"Linear: slope={slope:.4f}/day, R²={r_val**2:.3f}, p={p_linreg:.4f}")
        print(f"Permutation test (slope != 0): p={p_perm_slope:.4f}")

        # First vs Last day
        first_day = days_with_data[0]
        last_day = days_with_data[-1]
        first_smi = day_layer_data[first_day][layer]
        last_smi = day_layer_data[last_day][layer]

        _, p_fl = stats.mannwhitneyu(first_smi, last_smi, alternative='two-sided')
        delta, magnitude = cliffs_delta(last_smi, first_smi)
        change = medians_arr[-1] - medians_arr[0]

        print(f"First vs Last (Day {first_day} vs Day {last_day}): "
              f"delta={change:+.3f}, p={p_fl:.4f}, Cliff's delta={delta:.3f} ({magnitude})")

        results['by_layer'][layer] = {
            'days': days_with_data,
            'medians': medians,
            'n_cells': n_cells,
            'spearman': {'rho': rho, 'p': p_spearman, 'ci_95': rho_ci},
            'linreg': {'slope': slope, 'r2': r_val**2, 'p': p_linreg},
            'permutation_slope': {'p': p_perm_slope, 'observed_slope': slope},
            'first_vs_last': {'delta': change, 'p': p_fl,
                              'cliffs_delta': delta, 'effect_magnitude': magnitude}
        }

    return results


# ============================================================================
# ANALYSIS 3: LAYER DEVELOPMENT RATE COMPARISONS
# ============================================================================

def compare_layer_development_rates(all_data, temporal_results):
    """
    Q2b: Do layers develop SMI at different rates?

    Tests: Permutation tests comparing slopes between layers,
           Kruskal-Wallis on delta SMI (early Days 1-3 vs late Days 4+).
    """
    print(f"\n{'='*70}")
    print("ANALYSIS 3: LAYER DEVELOPMENT RATE COMPARISONS")
    print(f"{'='*70}")

    results = {'slope_comparisons': {}, 'delta_smi': {}}

    # Pairwise slope comparisons
    print("\nPairwise slope comparisons (permutation test):")
    for l1, l2 in combinations(LAYER_ORDER, 2):
        if l1 not in temporal_results['by_layer'] or l2 not in temporal_results['by_layer']:
            continue

        days1 = np.array(temporal_results['by_layer'][l1]['days'])
        medians1 = np.array(temporal_results['by_layer'][l1]['medians'])
        days2 = np.array(temporal_results['by_layer'][l2]['days'])
        medians2 = np.array(temporal_results['by_layer'][l2]['medians'])

        p_perm, obs_diff = permutation_compare_slopes(days1, medians1, days2, medians2,
                                                      n_permutations=10000)
        slope1 = temporal_results['by_layer'][l1]['linreg']['slope']
        slope2 = temporal_results['by_layer'][l2]['linreg']['slope']

        sig = '***' if p_perm < 0.001 else '**' if p_perm < 0.01 else '*' if p_perm < 0.05 else ''
        print(f"  {l1} vs {l2}: slopes {slope1:.4f} vs {slope2:.4f}, "
              f"diff={obs_diff:+.4f}, p={p_perm:.4f} {sig}")

        results['slope_comparisons'][f'{l1}_vs_{l2}'] = {
            'slope1': slope1, 'slope2': slope2, 'diff': obs_diff, 'p_permutation': p_perm
        }

    # Delta SMI analysis (early vs late per animal)
    early_days_list = [1, 2, 3]
    late_days_min = 4

    delta_smi_by_layer = {layer: [] for layer in LAYER_ORDER}

    for animal_id, animal_data in all_data.items():
        available_days = sorted(animal_data.keys())
        early_available = [d for d in available_days if d in early_days_list]
        late_available = [d for d in available_days if d >= late_days_min]

        if len(early_available) == 0 or len(late_available) == 0:
            continue

        for layer in LAYER_ORDER:
            early_smi = []
            for day in early_available:
                if layer in animal_data[day]['layers']:
                    early_smi.extend(animal_data[day]['layers'][layer]['SMI'])
            late_smi = []
            for day in late_available:
                if layer in animal_data[day]['layers']:
                    late_smi.extend(animal_data[day]['layers'][layer]['SMI'])

            if len(early_smi) > 0 and len(late_smi) > 0:
                delta_smi_by_layer[layer].append(np.median(late_smi) - np.median(early_smi))

    print("\nDelta SMI per layer (Late D4+ minus Early D1-3):")
    for layer in LAYER_ORDER:
        if len(delta_smi_by_layer[layer]) > 0:
            med_delta = np.median(delta_smi_by_layer[layer])
            n = len(delta_smi_by_layer[layer])
            print(f"  {layer}: median delta={med_delta:+.3f} (n={n} animals)")

    valid_layers_delta = [lyr for lyr in LAYER_ORDER if len(delta_smi_by_layer[lyr]) > 1]
    if len(valid_layers_delta) >= 2:
        groups_delta = [delta_smi_by_layer[lyr] for lyr in valid_layers_delta]
        h_stat, kw_p = stats.kruskal(*groups_delta)
        print(f"\nKruskal-Wallis (delta SMI across layers): H={h_stat:.3f}, p={kw_p:.4f}")
        results['delta_smi']['kruskal_wallis'] = {'H': h_stat, 'p': kw_p}
        results['delta_smi']['by_layer'] = delta_smi_by_layer

    return results


# ============================================================================
# ANALYSIS 4: EARLY vs LATE (Days 1-2 vs Days 6-7)
# ============================================================================

def analyze_early_vs_late_pooled(all_data):
    """
    Q3: Does SMI increase from early (Days 1-2) to late (Days 6-7)? (Pooled)

    Tests: Mann-Whitney U, Cliff's Delta effect size.
    """
    print(f"\n{'='*70}")
    print("ANALYSIS 4: EARLY (D1-2) vs LATE (D6-7) (POOLED ACROSS ANIMALS)")
    print(f"{'='*70}")

    results = {}

    for layer in LAYER_ORDER:
        early_smi = []
        late_smi = []
        early_animals = []
        late_animals = []

        for animal_id, animal_data in all_data.items():
            for day, session in animal_data.items():
                if layer in session['layers'] and len(session['layers'][layer]['SMI']) > 0:
                    smi_vals = session['layers'][layer]['SMI']
                    if day <= 2:
                        early_smi.extend(smi_vals)
                        early_animals.append(animal_id)
                    elif day >= 6:
                        late_smi.extend(smi_vals)
                        late_animals.append(animal_id)

        if len(early_smi) < 10 or len(late_smi) < 10:
            print(f"\n  {layer}: Insufficient data (early n={len(early_smi)}, late n={len(late_smi)})")
            continue

        early_smi = np.array(early_smi)
        late_smi = np.array(late_smi)

        u_stat, p_val = stats.mannwhitneyu(early_smi, late_smi, alternative='less')
        delta, magnitude = cliffs_delta(late_smi, early_smi)
        early_med = np.median(early_smi)
        late_med = np.median(late_smi)
        diff = late_med - early_med

        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        print(f"\n  {layer}:")
        print(f"    Early (n={len(early_smi)} from {len(set(early_animals))} animals): "
              f"median={early_med:.3f}")
        print(f"    Late  (n={len(late_smi)} from {len(set(late_animals))} animals): "
              f"median={late_med:.3f}")
        print(f"    delta={diff:+.3f}, p={p_val:.4f} {sig}, Cliff's delta={delta:.3f} ({magnitude})")

        results[layer] = {
            'early_median': early_med, 'late_median': late_med, 'delta': diff,
            'p_value': p_val, 'cliffs_delta': delta, 'effect_magnitude': magnitude,
            'early_n': len(early_smi), 'late_n': len(late_smi),
            'early_n_animals': len(set(early_animals)),
            'late_n_animals': len(set(late_animals))
        }

    return results


# ============================================================================
# ANALYSIS 5: GAP CLOSURE
# ============================================================================

def analyze_gap_closure_pooled(all_data):
    """
    Q4: Does the Deep - Superficial SMI gap narrow over time? (Pooled)

    Tests: Spearman correlation on gap vs day, linear regression.
    """
    print(f"\n{'='*70}")
    print("ANALYSIS 5: GAP CLOSURE (POOLED)")
    print(f"{'='*70}")

    all_days = sorted(set(d for a in all_data.values() for d in a.keys()))

    gaps = []
    gap_days = []
    sup_medians = []
    deep_medians = []

    for day in all_days:
        sup_smi = []
        deep_smi = []

        for animal_data in all_data.values():
            if day not in animal_data:
                continue
            session = animal_data[day]
            for layer in ['L2/3', 'L4']:
                if layer in session['layers']:
                    sup_smi.extend(session['layers'][layer]['SMI'])
            for layer in ['L5', 'L6']:
                if layer in session['layers']:
                    deep_smi.extend(session['layers'][layer]['SMI'])

        if len(sup_smi) > 0 and len(deep_smi) > 0:
            sup_med = np.median(sup_smi)
            deep_med = np.median(deep_smi)
            gap = deep_med - sup_med
            gap_days.append(day)
            gaps.append(gap)
            sup_medians.append(sup_med)
            deep_medians.append(deep_med)
            print(f"  Day {day}: Deep={deep_med:.3f}, Sup={sup_med:.3f}, Gap={gap:+.3f}")

    if len(gaps) < 3:
        print("  Insufficient data for gap trend analysis")
        return None

    gaps = np.array(gaps)
    gap_days = np.array(gap_days)

    rho, p_val = stats.spearmanr(gap_days, gaps)
    slope, _, _, _, _ = stats.linregress(gap_days, gaps)

    print(f"\n  Gap trend: rho={rho:.3f}, p={p_val:.4f}")
    print(f"  Linear: slope={slope:.4f}/day")
    print(f"  Initial gap: {gaps[0]:.3f}, Final gap: {gaps[-1]:.3f}")

    if rho < 0 and p_val < 0.05:
        print("  Gap SIGNIFICANTLY DECREASES (superficial catching up)")
    elif rho < 0:
        print("  Trend toward gap closure (not significant)")

    return {
        'days': gap_days.tolist(), 'gaps': gaps.tolist(),
        'sup_medians': sup_medians, 'deep_medians': deep_medians,
        'spearman_rho': rho, 'spearman_p': p_val,
        'slope': slope, 'initial_gap': gaps[0], 'final_gap': gaps[-1]
    }


# ============================================================================
# ANALYSIS 6: INDIVIDUAL ANIMAL TRAJECTORIES
# ============================================================================

def analyze_individual_trajectories(all_data):
    """Track each animal's SMI trajectory per layer."""

    print(f"\n{'='*70}")
    print("ANALYSIS 6: INDIVIDUAL ANIMAL TRAJECTORIES")
    print(f"{'='*70}")

    trajectories = {layer: {} for layer in LAYER_ORDER}

    for animal_id in sorted(all_data.keys()):
        animal_data = all_data[animal_id]

        for layer in LAYER_ORDER:
            days = []
            medians = []

            for day in sorted(animal_data.keys()):
                if layer in animal_data[day]['layers']:
                    ld = animal_data[day]['layers'][layer]
                    if ld['n_valid'] > 0:
                        days.append(day)
                        medians.append(ld['median_smi'])

            if len(days) >= 2:
                rho = stats.spearmanr(days, medians)[0] if len(days) >= 3 else np.nan
                trajectories[layer][animal_id] = {
                    'days': days, 'medians': medians,
                    'spearman_rho': rho,
                    'change': medians[-1] - medians[0]
                }

    print("\nTrajectory summary (first -> last day change):")
    for layer in LAYER_ORDER:
        if trajectories[layer]:
            changes = [t['change'] for t in trajectories[layer].values()
                       if not np.isnan(t['change'])]
            if changes:
                n_pos = sum(c > 0 for c in changes)
                print(f"  {layer}: {n_pos}/{len(changes)} animals show increase, "
                      f"mean delta={np.mean(changes):+.3f}")

    return trajectories


# ============================================================================
# VISUALIZATION: 3x3 FIGURE
# ============================================================================

def create_across_animals_visualizations(all_data, day1_results, temporal_results,
                                         development_results, early_late_results,
                                         gap_results, trajectory_results, save_path=None):
    """
    3x3 publication figure:
    Row 1: SMI over time (line+SEM), heatmap, Day 1 violin
    Row 2: Early vs Late bars, Sup vs Deep trajectory, Gap closure
    Row 3: Development slopes, Individual trajectories (L2/3), Statistical summary
    """
    fig = plt.figure(figsize=(22, 18))
    all_days = sorted(set(d for a in all_data.values() for d in a.keys()))

    # =========================================================================
    # Panel 1: SMI over time with SEM
    # =========================================================================
    ax1 = fig.add_subplot(3, 3, 1)

    for layer in LAYER_ORDER:
        days_plot, medians_plot, sems_plot = [], [], []
        for day in all_days:
            all_smi = []
            for animal_data in all_data.values():
                if day in animal_data and layer in animal_data[day]['layers']:
                    smi_vals = animal_data[day]['layers'][layer]['SMI']
                    if len(smi_vals) > 0:
                        all_smi.extend(smi_vals)
            if len(all_smi) > 0:
                days_plot.append(day)
                medians_plot.append(np.median(all_smi))
                sems_plot.append(stats.sem(all_smi))

        if not days_plot:
            continue
        days_plot = np.array(days_plot)
        medians_plot = np.array(medians_plot)
        sems_plot = np.array(sems_plot)

        ax1.plot(days_plot, medians_plot, 'o-', color=LAYER_COLORS[layer], linewidth=3,
                 markersize=10, label=layer, zorder=3, markeredgecolor='white',
                 markeredgewidth=1.5)
        ax1.fill_between(days_plot, medians_plot - sems_plot, medians_plot + sems_plot,
                         color=LAYER_COLORS[layer], alpha=0.3, zorder=2)

    ax1.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Median SMI', fontsize=12, fontweight='bold')
    ax1.set_title('SMI Development Over Time\n(shaded = SEM, pooled cells)',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # =========================================================================
    # Panel 2: Heatmap (Days x Layers) with cell + animal counts
    # =========================================================================
    ax2 = fig.add_subplot(3, 3, 2)

    heatmap = np.full((len(LAYER_ORDER), len(all_days)), np.nan)
    n_cells_matrix = np.zeros((len(LAYER_ORDER), len(all_days)), dtype=int)
    n_animals_matrix = np.zeros((len(LAYER_ORDER), len(all_days)), dtype=int)

    for j, day in enumerate(all_days):
        for i, layer in enumerate(LAYER_ORDER):
            all_smi = []
            animal_count = 0
            for animal_data in all_data.values():
                if day in animal_data and layer in animal_data[day]['layers']:
                    smi_vals = animal_data[day]['layers'][layer]['SMI']
                    if len(smi_vals) > 0:
                        all_smi.extend(smi_vals)
                        animal_count += 1
            if len(all_smi) > 0:
                heatmap[i, j] = np.median(all_smi)
                n_cells_matrix[i, j] = len(all_smi)
                n_animals_matrix[i, j] = animal_count

    im = ax2.imshow(heatmap, cmap='YlOrRd', aspect='auto', vmin=0,
                    vmax=np.nanmax(heatmap) if not np.all(np.isnan(heatmap)) else 1)
    ax2.set_xticks(range(len(all_days)))
    ax2.set_xticklabels([f'D{d}' for d in all_days])
    ax2.set_yticks(range(len(LAYER_ORDER)))
    ax2.set_yticklabels(LAYER_ORDER)

    for i in range(len(LAYER_ORDER)):
        for j in range(len(all_days)):
            if not np.isnan(heatmap[i, j]):
                text_color = 'white' if heatmap[i, j] > np.nanmax(heatmap) * 0.6 else 'black'
                ax2.text(j, i, f'{heatmap[i, j]:.2f}\nn={n_cells_matrix[i, j]}\n'
                         f'({n_animals_matrix[i, j]} mice)',
                         ha='center', va='center', fontsize=7.5, color=text_color)

    ax2.set_title('Pooled Median SMI\n(cells from all animals)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Median SMI', shrink=0.8)

    # =========================================================================
    # Panel 3: Day 1 violin plot
    # =========================================================================
    ax3 = fig.add_subplot(3, 3, 3)

    pooled = day1_results['pooled_smi']
    plot_data, plot_labels, plot_colors, positions = [], [], [], []

    for idx, layer in enumerate(LAYER_ORDER):
        if len(pooled[layer]) > 0:
            plot_data.append(pooled[layer])
            plot_labels.append(f'{layer}\n(n={len(pooled[layer])})')
            plot_colors.append(LAYER_COLORS[layer])
            positions.append(idx)

    if plot_data:
        parts = ax3.violinplot(plot_data, positions=positions, widths=0.7,
                               showmeans=False, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plot_colors[i])
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        parts['cmedians'].set_edgecolor('black')
        parts['cmedians'].set_linewidth(2)

        for i, data in enumerate(plot_data):
            data = np.array(data)
            y = data[np.random.choice(len(data), size=min(500, len(data)), replace=False)]
            x = np.random.normal(positions[i], 0.04, size=len(y))
            ax3.scatter(x, y, alpha=0.2, s=1.5, color=plot_colors[i])

        if 'kruskal_wallis' in day1_results:
            kw_p = day1_results['kruskal_wallis']['p']
            sig = '***' if kw_p < 0.001 else '**' if kw_p < 0.01 else '*' if kw_p < 0.05 else 'ns'
            ax3.text(0.97, 0.97, f'KW p={kw_p:.2e} {sig}', transform=ax3.transAxes,
                     ha='right', va='top', fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.8))

        ax3.set_xticks(positions)
        ax3.set_xticklabels(plot_labels)
        ax3.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax3.set_ylim([-0.1, 1.0])
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

    ax3.set_ylabel('SMI', fontsize=12, fontweight='bold')
    ax3.set_title('Day 1: Layer Comparison\n(pooled across animals)', fontsize=12, fontweight='bold')

    # =========================================================================
    # Panel 4: Early (D1-2) vs Late (D6-7)
    # =========================================================================
    ax4 = fig.add_subplot(3, 3, 4)

    if early_late_results:
        x = np.arange(len(LAYER_ORDER))
        width = 0.35

        early_vals = [early_late_results.get(lyr, {}).get('early_median', np.nan) for lyr in LAYER_ORDER]
        late_vals = [early_late_results.get(lyr, {}).get('late_median', np.nan) for lyr in LAYER_ORDER]

        ax4.bar(x - width / 2, early_vals, width, label='Early (D1-2)',
                color='steelblue', alpha=0.7, edgecolor='black')
        ax4.bar(x + width / 2, late_vals, width, label='Late (D6-7)',
                color='coral', alpha=0.7, edgecolor='black')

        for i, layer in enumerate(LAYER_ORDER):
            if layer in early_late_results:
                p = early_late_results[layer]['p_value']
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                if sig:
                    max_val = max(early_vals[i] or 0, late_vals[i] or 0)
                    ax4.text(i, max_val + 0.02, sig, ha='center', fontsize=12,
                             fontweight='bold')

        ax4.set_xticks(x)
        ax4.set_xticklabels(LAYER_ORDER)
        ax4.legend(fontsize=10)

    ax4.set_ylabel('Median SMI', fontsize=12, fontweight='bold')
    ax4.set_title('Early vs Late Development\n(D1-2 vs D6-7)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Panel 5: Superficial vs Deep trajectory
    # =========================================================================
    ax5 = fig.add_subplot(3, 3, 5)

    sup_by_day = defaultdict(list)
    deep_by_day = defaultdict(list)

    for animal_data in all_data.values():
        for day, session in animal_data.items():
            for layer in ['L2/3', 'L4']:
                if layer in session['layers'] and len(session['layers'][layer]['SMI']) > 0:
                    sup_by_day[day].extend(session['layers'][layer]['SMI'])
            for layer in ['L5', 'L6']:
                if layer in session['layers'] and len(session['layers'][layer]['SMI']) > 0:
                    deep_by_day[day].extend(session['layers'][layer]['SMI'])

    days_common = sorted(set(sup_by_day.keys()) & set(deep_by_day.keys()))
    if days_common:
        sup_meds = [np.median(sup_by_day[d]) for d in days_common]
        deep_meds = [np.median(deep_by_day[d]) for d in days_common]
        sup_sems = [stats.sem(sup_by_day[d]) for d in days_common]
        deep_sems = [stats.sem(deep_by_day[d]) for d in days_common]

        ax5.plot(days_common, sup_meds, 'o-', color='#1E88E5', linewidth=3,
                 markersize=12, label='Superficial (L2/3+L4)')
        ax5.fill_between(days_common,
                         np.array(sup_meds) - np.array(sup_sems),
                         np.array(sup_meds) + np.array(sup_sems),
                         color='#1E88E5', alpha=0.2)
        ax5.plot(days_common, deep_meds, 's-', color='#E53935', linewidth=3,
                 markersize=12, label='Deep (L5+L6)')
        ax5.fill_between(days_common,
                         np.array(deep_meds) - np.array(deep_sems),
                         np.array(deep_meds) + np.array(deep_sems),
                         color='#E53935', alpha=0.2)
        ax5.fill_between(days_common, sup_meds, deep_meds, alpha=0.1, color='gray')

    ax5.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Median SMI', fontsize=12, fontweight='bold')
    ax5.set_title('Superficial vs Deep Layers\n(shaded = SEM)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10, framealpha=0.9)
    ax5.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 6: Gap closure (Deep - Superficial over time)
    # =========================================================================
    ax6 = fig.add_subplot(3, 3, 6)

    if gap_results:
        ax6.plot(gap_results['days'], gap_results['gaps'], 'ko-', linewidth=2, markersize=10)
        ax6.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax6.fill_between(gap_results['days'], 0, gap_results['gaps'], alpha=0.3, color='purple')

        z = np.polyfit(gap_results['days'], gap_results['gaps'], 1)
        p = np.poly1d(z)
        ax6.plot(gap_results['days'], p(gap_results['days']), 'r--', alpha=0.7,
                 label=f"slope={gap_results['slope']:.3f}/day")

        ax6.text(0.97, 0.97,
                 f"rho={gap_results['spearman_rho']:.2f}\np={gap_results['spearman_p']:.3f}",
                 transform=ax6.transAxes, ha='right', va='top', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8))
        ax6.legend(loc='lower left', fontsize=9)

        # Annotate initial and final gap
        ax6.annotate(f"Gap={gap_results['initial_gap']:.2f}",
                     xy=(gap_results['days'][0],
                         (gap_results['sup_medians'][0] + gap_results['deep_medians'][0]) / 2),
                     fontsize=9)
        ax6.annotate(f"Gap={gap_results['final_gap']:.2f}",
                     xy=(gap_results['days'][-1],
                         (gap_results['sup_medians'][-1] + gap_results['deep_medians'][-1]) / 2),
                     fontsize=9, ha='right')

    ax6.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Gap (Deep - Superficial)', fontsize=12, fontweight='bold')
    ax6.set_title('Gap Closure Analysis', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 7: Development slopes with bootstrap CI
    # =========================================================================
    ax7 = fig.add_subplot(3, 3, 7)

    slopes = []
    slope_cis = []
    slope_labels = []
    slope_colors = []

    for layer in LAYER_ORDER:
        if layer in temporal_results['by_layer']:
            slope = temporal_results['by_layer'][layer]['linreg']['slope']
            slopes.append(slope)
            slope_labels.append(layer)
            slope_colors.append(LAYER_COLORS[layer])

            days = np.array(temporal_results['by_layer'][layer]['days'])
            meds = np.array(temporal_results['by_layer'][layer]['medians'])

            boot_slopes = []
            for _ in range(5000):
                indices = np.random.choice(len(days), size=len(days), replace=True)
                boot_slope, _, _, _, _ = stats.linregress(days[indices], meds[indices])
                boot_slopes.append(boot_slope)

            slope_cis.append((np.percentile(boot_slopes, 2.5),
                              np.percentile(boot_slopes, 97.5)))

    if slopes:
        x_pos = np.arange(len(slopes))
        yerr_lower = [slopes[i] - slope_cis[i][0] for i in range(len(slopes))]
        yerr_upper = [slope_cis[i][1] - slopes[i] for i in range(len(slopes))]

        ax7.bar(x_pos, slopes, color=slope_colors, alpha=0.7, edgecolor='black',
                linewidth=1.5, yerr=[yerr_lower, yerr_upper], capsize=5,
                error_kw={'linewidth': 2})

        for i, layer in enumerate(slope_labels):
            p_val = temporal_results['by_layer'][layer]['permutation_slope']['p']
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            y_pos = slopes[i] + yerr_upper[i] + 0.002
            ax7.text(i, y_pos, sig, ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax7.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels(slope_labels)

    ax7.set_ylabel('Slope (dSMI/day)', fontsize=12, fontweight='bold')
    ax7.set_title('Development Rate by Layer\n(95% CI, * p<0.05, ** p<0.01, *** p<0.001)',
                  fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)

    # =========================================================================
    # Panel 8: Individual animal trajectories (L2/3)
    # =========================================================================
    ax8 = fig.add_subplot(3, 3, 8)

    if 'L2/3' in trajectory_results:
        for animal_id, t in trajectory_results['L2/3'].items():
            ax8.plot(t['days'], t['medians'], 'o-', alpha=0.5, linewidth=1.5, label=animal_id)

        if 'L2/3' in temporal_results['by_layer']:
            tr = temporal_results['by_layer']['L2/3']
            ax8.plot(tr['days'], tr['medians'], 'k-', linewidth=3, label='Pooled', zorder=10)

    ax8.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Median SMI', fontsize=12, fontweight='bold')
    ax8.set_title('L2/3: Individual Trajectories', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=8, loc='lower right')
    ax8.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 9: Statistical summary text
    # =========================================================================
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')

    n_animals = len(all_data)
    summary = f"STATISTICAL SUMMARY\n{'='*38}\n\n"
    summary += f"Animals: {n_animals}\n"
    summary += f"Days: {min(all_days)} to {max(all_days)}\n\n"

    summary += "DAY 1 LAYER DIFFERENCES:\n"
    if 'kruskal_wallis' in day1_results:
        kw_p = day1_results['kruskal_wallis']['p']
        sig = '***' if kw_p < 0.001 else '**' if kw_p < 0.01 else '*' if kw_p < 0.05 else ''
        summary += f"  KW: p={kw_p:.2e} {sig}\n"
    if 'superficial_vs_deep' in day1_results:
        svd = day1_results['superficial_vs_deep']
        summary += (f"  Deep>Sup: p={svd['p_permutation']:.4f}, "
                    f"delta={svd['cliffs_delta']:.2f} ({svd['effect_magnitude']})\n")

    summary += "\nEARLY vs LATE (D1-2 vs D6-7):\n"
    for layer in LAYER_ORDER:
        if layer in early_late_results:
            el = early_late_results[layer]
            sig = '*' if el['p_value'] < 0.05 else ''
            summary += (f"  {layer}: delta={el['delta']:+.3f}, p={el['p_value']:.3f}, "
                        f"delta_cliff={el['cliffs_delta']:.2f} {sig}\n")

    summary += "\nTEMPORAL SLOPES:\n"
    for layer in LAYER_ORDER:
        if layer in temporal_results['by_layer']:
            tr = temporal_results['by_layer'][layer]
            p_s = tr['permutation_slope']['p']
            sig = '*' if p_s < 0.05 else ''
            summary += (f"  {layer}: {tr['linreg']['slope']:.4f}/day, "
                        f"p={p_s:.3f} {sig}\n")

    if gap_results:
        summary += "\nGAP CLOSURE:\n"
        summary += (f"  Initial: {gap_results['initial_gap']:.3f}, "
                    f"Final: {gap_results['final_gap']:.3f}\n")
        summary += (f"  rho={gap_results['spearman_rho']:.2f}, "
                    f"p={gap_results['spearman_p']:.3f}\n")

    ax9.text(0.05, 0.97, summary, transform=ax9.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Across-Animals SMI Analysis', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_path:
        # Mega figure
        fig_path = os.path.join(save_path, 'across_animals_smi_analysis.png')
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f"\nSaved figure: {fig_path}")

        # Individual panels
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        panel_axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
        panel_names = [
            'smi_trajectory', 'smi_heatmap', 'day1_violin',
            'early_vs_late', 'sup_vs_deep', 'gap_closure',
            'slopes', 'L23_trajectories', 'summary',
        ]
        for ax, name in zip(panel_axes, panel_names):
            try:
                bbox = ax.get_tightbbox(renderer)
                bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
                out = os.path.join(save_path, f'across_animals_panel_{name}.png')
                fig.savefig(out, bbox_inches=bbox.expanded(1.15, 1.15), dpi=200)
                print(f'  Saved: across_animals_panel_{name}.png')
            except Exception:
                pass

    return fig


# ============================================================================
# SUMMARY TABLES
# ============================================================================

def create_summary_table(all_data, temporal_results, save_path=None):
    """Create and save per-session and per-layer slope summary tables."""
    rows = []

    for animal_id in sorted(all_data.keys()):
        for day in sorted(all_data[animal_id].keys()):
            session = all_data[animal_id][day]
            row = {'Animal': animal_id, 'Day': day}

            for layer in LAYER_ORDER:
                if layer in session['layers']:
                    ld = session['layers'][layer]
                    row[f'{layer}_n'] = ld['n_valid']
                    row[f'{layer}_median'] = round(ld['median_smi'], 4)
                    prop = ld.get('prop_modulated', np.nan)
                    row[f'{layer}_prop_mod'] = round(prop, 4) if not np.isnan(prop) else np.nan
                else:
                    row[f'{layer}_n'] = 0
                    row[f'{layer}_median'] = np.nan
                    row[f'{layer}_prop_mod'] = np.nan

            rows.append(row)

    df = pd.DataFrame(rows)

    slope_rows = []
    for layer in LAYER_ORDER:
        if layer in temporal_results['by_layer']:
            tr = temporal_results['by_layer'][layer]
            slope_rows.append({
                'Layer': layer,
                'Slope (dSMI/day)': tr['linreg']['slope'],
                'R2': tr['linreg']['r2'],
                'Spearman rho': tr['spearman']['rho'],
                'Spearman p': tr['spearman']['p'],
                'Permutation p (slope)': tr['permutation_slope']['p']
            })

    df_slopes = pd.DataFrame(slope_rows)

    if save_path:
        csv_path = os.path.join(save_path, 'smi_summary_table.csv')
        df.to_csv(csv_path, index=False)

        slope_path = os.path.join(save_path, 'smi_slopes_table.csv')
        df_slopes.to_csv(slope_path, index=False)

        print(f"Saved: {csv_path}")
        print(f"Saved: {slope_path}")

    return df, df_slopes


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def run_across_animals_analysis(parent_dir, save_path=None):
    """Complete workflow for across-animals SMI analysis."""
    print(f"\n{'='*80}")
    print("ACROSS-ANIMALS SMI ANALYSIS")
    print(f"{'='*80}")

    all_data = load_all_animals_smi(parent_dir)

    if len(all_data) == 0:
        print("No data loaded. Exiting.")
        return None

    if save_path is None:
        save_path = os.path.join(parent_dir, 'across_animals_smi_analysis')
    os.makedirs(save_path, exist_ok=True)

    day1_results = analyze_day1_layer_differences_pooled(all_data)
    temporal_results = analyze_temporal_progression_pooled(all_data)
    development_results = compare_layer_development_rates(all_data, temporal_results)
    early_late_results = analyze_early_vs_late_pooled(all_data)
    gap_results = analyze_gap_closure_pooled(all_data)
    trajectory_results = analyze_individual_trajectories(all_data)

    fig = create_across_animals_visualizations(
        all_data, day1_results, temporal_results, development_results,
        early_late_results, gap_results, trajectory_results, save_path
    )

    summary_df, slopes_df = create_summary_table(all_data, temporal_results, save_path)

    print(f"\n{'='*70}")
    print("SUMMARY TABLE (first 10 rows)")
    print(f"{'='*70}")
    print(summary_df.head(10).to_string(index=False))

    print(f"\n{'='*70}")
    print("SLOPES TABLE")
    print(f"{'='*70}")
    print(slopes_df.to_string(index=False))

    results = {
        'all_data': all_data,
        'day1_layer_differences': day1_results,
        'temporal_progression': temporal_results,
        'development_comparison': development_results,
        'early_vs_late': early_late_results,
        'gap_closure': gap_results,
        'individual_trajectories': trajectory_results,
        'summary_table': summary_df,
        'slopes_table': slopes_df,
        'figure': fig,
        'save_path': save_path
    }

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Outputs saved to: {save_path}")
    print("  - across_animals_smi_analysis.png")
    print("  - smi_summary_table.csv")
    print("  - smi_slopes_table.csv")

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parent_dir = r"D:\V1_SpatialModulation\2p\V1_prism"
    save_dir = r"D:\V1_SpatialModulation\2p\V1_prism\across_animals_smi_analysis"

    results = run_across_animals_analysis(parent_dir, save_dir)

    plt.show()
