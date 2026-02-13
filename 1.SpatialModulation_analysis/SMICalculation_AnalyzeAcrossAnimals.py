"""
SMICalculation_AnalyzeAcrossAnimals.py

Compare SMI development across multiple animals.

Analyzes (pooled across animals):
1. Layer differences within Day 1 - Do deeper layers start with higher SMI?
2. Temporal progression per layer - Does L2/3 develop SMI over days?
3. Animal-specific patterns vs population trends

Updated with robust non-parametric statistics and enhanced visualizations.

Input: SMI result files from all animals
Output: Statistical comparisons, pooled visualizations, summary tables

JSY, 2025
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import h5py
from glob import glob
from collections import defaultdict
import pandas as pd
from itertools import combinations

# ============================================================================
# STATISTICAL UTILITY FUNCTIONS
# ============================================================================

def cliffs_delta(x, y):
    """
    Calculate Cliff's Delta effect size.
    
    Cliff's Delta is a non-parametric effect size measure:
    - Values range from -1 to 1
    - |delta| < 0.147 = negligible
    - |delta| < 0.33 = small
    - |delta| < 0.474 = medium
    - |delta| >= 0.474 = large
    
    Returns:
        delta (float): Cliff's Delta
        magnitude (str): Interpretation of effect size
    """
    if len(x) == 0 or len(y) == 0:
        return np.nan, "undefined"
    
    # Calculate dominance matrix
    n_x, n_y = len(x), len(y)
    greater = np.sum([np.sum(x_i > y) for x_i in x])
    less = np.sum([np.sum(x_i < y) for x_i in x])
    
    delta = (greater - less) / (n_x * n_y)
    
    # Interpret magnitude
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
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Parameters:
        data (array): Input data (1D or 2D for paired data)
        statistic (function): Function to compute (default: median)
        n_bootstrap (int): Number of bootstrap samples
        ci (float): Confidence interval percentage (e.g., 95)
        paired (bool): If True, resample rows (for paired/multivariate data)
    
    Returns:
        (lower, upper): Confidence interval bounds
    """
    if len(data) == 0:
        return (np.nan, np.nan)
    
    data = np.array(data)
    bootstrap_stats = []
    
    if paired and data.ndim == 2:
        # For paired/multivariate data, resample rows
        n_samples = data.shape[0]
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_sample = data[indices]
            bootstrap_stats.append(statistic(bootstrap_sample))
    else:
        # For univariate data
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic(bootstrap_sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile
    
    lower = np.percentile(bootstrap_stats, lower_percentile)
    upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return (lower, upper)



def permutation_test(group1, group2, n_permutations=10000, statistic='median_diff'):
    """
    Permutation test for difference between two groups.
    
    Parameters:
        group1, group2 (arrays): Data for two groups
        n_permutations (int): Number of permutations
        statistic (str): 'median_diff' or 'mean_diff'
    
    Returns:
        p_value (float): Two-tailed p-value
        observed_stat (float): Observed test statistic
    """
    if len(group1) == 0 or len(group2) == 0:
        return np.nan, np.nan
    
    # Observed statistic
    if statistic == 'median_diff':
        observed_stat = np.median(group1) - np.median(group2)
    else:
        observed_stat = np.mean(group1) - np.mean(group2)
    
    # Combine data
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    # Permutation distribution
    perm_stats = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        
        if statistic == 'median_diff':
            perm_stat = np.median(perm_group1) - np.median(perm_group2)
        else:
            perm_stat = np.mean(perm_group1) - np.mean(perm_group2)
        
        perm_stats.append(perm_stat)
    
    perm_stats = np.array(perm_stats)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
    
    return p_value, observed_stat


def fdr_correction(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction for multiple comparisons.
    
    Returns:
        reject (array): Boolean array indicating which hypotheses to reject
        p_adjusted (array): Adjusted p-values
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # BH procedure
    reject = np.zeros(n, dtype=bool)
    for i, p in enumerate(sorted_p):
        if p <= (i + 1) / n * alpha:
            reject[sorted_indices[i]] = True
    
    # Adjusted p-values
    p_adjusted = np.minimum(sorted_p * n / np.arange(1, n + 1), 1.0)
    p_adjusted = np.minimum.accumulate(p_adjusted[::-1])[::-1]
    
    # Restore original order
    original_order = np.argsort(sorted_indices)
    p_adjusted = p_adjusted[original_order]
    
    return reject, p_adjusted


def permutation_slope_test(days, medians, n_permutations=10000):
    """
    Permutation test for slope significance using linear regression.
    
    Tests H0: slope = 0 (no temporal trend)
    
    Returns:
        p_value (float): Two-tailed p-value
        observed_slope (float): Observed slope from linear regression
    """
    if len(days) < 2:
        return np.nan, np.nan
    
    # Observed slope
    slope, intercept, _, _, _ = stats.linregress(days, medians)
    observed_slope = slope
    
    # Permutation distribution
    perm_slopes = []
    for _ in range(n_permutations):
        perm_medians = np.random.permutation(medians)
        perm_slope, _, _, _, _ = stats.linregress(days, perm_medians)
        perm_slopes.append(perm_slope)
    
    perm_slopes = np.array(perm_slopes)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_slopes) >= np.abs(observed_slope))
    
    return p_value, observed_slope


def permutation_compare_slopes(days1, medians1, days2, medians2, n_permutations=10000):
    """
    Permutation test comparing slopes between two groups (e.g., two layers).
    
    Tests H0: slope1 = slope2
    
    Returns:
        p_value (float): Two-tailed p-value
        observed_diff (float): Observed difference in slopes
    """
    if len(days1) < 2 or len(days2) < 2:
        return np.nan, np.nan
    
    # Observed slopes
    slope1, _, _, _, _ = stats.linregress(days1, medians1)
    slope2, _, _, _, _ = stats.linregress(days2, medians2)
    observed_diff = slope1 - slope2
    
    # Combine data for permutation
    all_days = np.concatenate([days1, days2])
    all_medians = np.concatenate([medians1, medians2])
    all_labels = np.concatenate([np.zeros(len(days1)), np.ones(len(days2))])
    
    # Permutation distribution
    perm_diffs = []
    for _ in range(n_permutations):
        perm_labels = np.random.permutation(all_labels)
        
        perm_group1_idx = perm_labels == 0
        perm_group2_idx = perm_labels == 1
        
        if np.sum(perm_group1_idx) >= 2 and np.sum(perm_group2_idx) >= 2:
            perm_slope1, _, _, _, _ = stats.linregress(all_days[perm_group1_idx], 
                                                        all_medians[perm_group1_idx])
            perm_slope2, _, _, _, _ = stats.linregress(all_days[perm_group2_idx], 
                                                        all_medians[perm_group2_idx])
            perm_diffs.append(perm_slope1 - perm_slope2)
    
    perm_diffs = np.array(perm_diffs)
    
    # Two-tailed p-value
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
    """Load SMI data from all animals."""
    print(f"\n{'='*70}")
    print("LOADING ALL ANIMALS SMI DATA")
    print(f"{'='*70}")
    print(f"Parent directory: {parent_dir}\n")
    
    smi_files = glob(os.path.join(parent_dir, "**", smi_pattern), recursive=True)
    print(f"Found {len(smi_files)} SMI result files\n")
    
    all_data = defaultdict(lambda: defaultdict(dict))  # {animal: {day: data}}
    
    for smi_path in sorted(smi_files):
        animal_id, day = extract_info(smi_path)
        if animal_id is None or day is None:
            continue
        
        print(f"  Loading {animal_id} Day {day}")
        
        try:
            with h5py.File(smi_path, 'r') as f:
                session_data = {'day': day, 'layers': {}}
                
                # Global stats
                if 'global_smi' in f:
                    g = f['global_smi']
                    session_data['global_median'] = g.attrs.get('median_smi', np.nan)
                    session_data['global_n'] = g.attrs.get('n_valid_cells', 0)
                
                # Layer data
                if 'layer_smi' in f:
                    for layer_key in f['layer_smi'].keys():
                        lg = f['layer_smi'][layer_key]
                        layer_name = lg.attrs.get('original_name', layer_key.replace('_', '/'))
                        smi_vals = lg['SMI'][:] if 'SMI' in lg else np.array([])
                        
                        session_data['layers'][layer_name] = {
                            'n_valid': lg.attrs.get('n_cells_valid', len(smi_vals)),
                            'median_smi': lg.attrs.get('median_smi', np.nan),
                            'mean_smi': lg.attrs.get('mean_smi', np.nan),
                            'mad_smi': lg.attrs.get('mad_smi', np.nan),
                            'SMI': smi_vals
                        }
                
                all_data[animal_id][day] = session_data
                
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Summary
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
    Question 1: On Day 1, do deeper layers have higher SMI? (Pooled across animals)
    
    Statistical tests:
    - Kruskal-Wallis (overall layer difference)
    - Mann-Whitney U pairwise comparisons with FDR correction
    - Cliff's Delta effect sizes
    - Permutation test for Superficial vs Deep
    """
    print(f"\n{'='*70}")
    print("ANALYSIS 1: DAY 1 LAYER DIFFERENCES (POOLED)")
    print(f"{'='*70}")
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    
    # Collect all Day 1 SMI values per layer
    pooled_smi = {layer: [] for layer in layer_order}
    animal_medians = {layer: [] for layer in layer_order}
    
    for animal_id, animal_data in all_data.items():
        day1 = min(animal_data.keys())  # First available day
        if day1 > 2:  # Skip if no early data
            continue
        
        session = animal_data[day1]
        for layer in layer_order:
            if layer in session['layers'] and len(session['layers'][layer]['SMI']) > 0:
                smi_vals = session['layers'][layer]['SMI']
                pooled_smi[layer].extend(smi_vals)
                animal_medians[layer].append(session['layers'][layer]['median_smi'])
    
    # Print summary
    print("\nPooled SMI values (Day 1):")
    for layer in layer_order:
        if len(pooled_smi[layer]) > 0:
            med = np.median(pooled_smi[layer])
            mad = stats.median_abs_deviation(pooled_smi[layer])
            n = len(pooled_smi[layer])
            n_animals = len(animal_medians[layer])
            print(f"  {layer}: n={n} cells from {n_animals} animals, median={med:.3f}±{mad:.3f}")
    
    # Statistical tests
    valid_layers = [l for l in layer_order if len(pooled_smi[l]) > 0]
    results = {'pooled_smi': pooled_smi, 'animal_medians': animal_medians}
    
    if len(valid_layers) >= 2:
        # ====================================================================
        # 1. Kruskal-Wallis (overall test)
        # ====================================================================
        groups = [pooled_smi[l] for l in valid_layers]
        h_stat, kw_p = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis (pooled cells): H={h_stat:.3f}, p={kw_p:.6f}")
        results['kruskal_wallis'] = {'H': h_stat, 'p': kw_p}
        
        # ====================================================================
        # 2. Pairwise Mann-Whitney with FDR correction
        # ====================================================================
        print("\nPairwise comparisons (Mann-Whitney U):")
        pairwise = {}
        p_values_list = []
        comparison_names = []
        
        for i, l1 in enumerate(valid_layers):
            for l2 in valid_layers[i+1:]:
                u_stat, p_val = stats.mannwhitneyu(pooled_smi[l1], pooled_smi[l2], 
                                                    alternative='two-sided')
                
                # Cliff's Delta effect size
                delta, magnitude = cliffs_delta(pooled_smi[l1], pooled_smi[l2])
                
                comparison_names.append(f'{l1}_vs_{l2}')
                p_values_list.append(p_val)
                
                pairwise[f'{l1}_vs_{l2}'] = {
                    'U': u_stat, 
                    'p': p_val,
                    'cliffs_delta': delta,
                    'effect_magnitude': magnitude
                }
        
        # FDR correction
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
            
            print(f"  {comp_name}: p={p_raw:.6f}{sig_raw}, p_FDR={p_fdr:.6f}{sig_fdr}, δ={delta:.3f} ({mag})")
        
        results['pairwise'] = pairwise
        
        # ====================================================================
        # 3. Superficial vs Deep (permutation test)
        # ====================================================================
        if 'L2/3' in valid_layers and ('L5' in valid_layers or 'L6' in valid_layers):
            sup_smi = pooled_smi['L2/3']
            deep_smi = []
            for l in ['L5', 'L6']:
                if l in valid_layers:
                    deep_smi.extend(pooled_smi[l])
            
            # Mann-Whitney U
            u_stat, p_mw = stats.mannwhitneyu(sup_smi, deep_smi, alternative='two-sided')
            
            # Permutation test
            p_perm, obs_diff = permutation_test(deep_smi, sup_smi, n_permutations=10000)
            
            # Cliff's Delta
            delta, magnitude = cliffs_delta(deep_smi, sup_smi)
            
            # Bootstrap CI for median difference
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
            print(f"  Cliff's Delta: δ={delta:.3f} ({magnitude})")
            print(f"  95% CI (median diff): [{ci_lower:.3f}, {ci_upper:.3f}]")
            
            if p_perm < 0.05 and np.median(deep_smi) > np.median(sup_smi):
                print("  ✓ Deep layers have significantly HIGHER SMI on Day 1")
            else:
                print("  → No significant difference or opposite pattern")
            
            results['superficial_vs_deep'] = {
                'p_mannwhitney': p_mw,
                'p_permutation': p_perm,
                'observed_diff': obs_diff,
                'cliffs_delta': delta,
                'effect_magnitude': magnitude,
                'ci_95': (ci_lower, ci_upper)
            }
    
    return results


# ============================================================================
# ANALYSIS 2: TEMPORAL PROGRESSION
# ============================================================================

def analyze_temporal_progression_pooled(all_data):
    """
    Question 2: Does SMI increase over days in each layer? (Pooled)
    
    Statistical tests:
    - Spearman correlation (day vs median SMI)
    - Linear regression + permutation test for slope significance
    - First vs Last day Mann-Whitney U
    - Bootstrap CI for correlation
    """
    print(f"\n{'='*70}")
    print("ANALYSIS 2: TEMPORAL PROGRESSION (POOLED)")
    print(f"{'='*70}")
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    
    # Organize data by day and layer
    day_layer_data = defaultdict(lambda: {l: [] for l in layer_order})
    all_days = set()
    
    for animal_id, animal_data in all_data.items():
        for day, session in animal_data.items():
            all_days.add(day)
            for layer in layer_order:
                if layer in session['layers'] and len(session['layers'][layer]['SMI']) > 0:
                    day_layer_data[day][layer].extend(session['layers'][layer]['SMI'])
    
    days_sorted = sorted(all_days)
    results = {'by_layer': {}}
    
    for layer in layer_order:
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
        
        # ====================================================================
        # 1. Spearman correlation
        # ====================================================================
        rho, p_spearman = stats.spearmanr(days_arr, medians_arr)
        
        # Bootstrap CI for Spearman rho
        def spearman_stat(data):
            return stats.spearmanr(data[:, 0], data[:, 1])[0]
        
        data_combined = np.column_stack([days_arr, medians_arr])
        rho_ci = bootstrap_ci(data_combined, statistic=spearman_stat, 
                             n_bootstrap=5000, paired=True)  # <-- Add paired=True
        
        print(f"\nSpearman correlation: ρ={rho:.3f}, p={p_spearman:.4f}")
        print(f"  95% CI: [{rho_ci[0]:.3f}, {rho_ci[1]:.3f}]")
        
        # ====================================================================
        # 2. Linear regression + permutation test
        # ====================================================================
        slope, intercept, r_val, p_linreg, _ = stats.linregress(days_arr, medians_arr)
        
        # Permutation test for slope
        p_perm_slope, obs_slope = permutation_slope_test(days_arr, medians_arr, n_permutations=10000)
        
        print(f"\nLinear regression:")
        print(f"  Slope: {slope:.4f}/day, R²={r_val**2:.3f}, p={p_linreg:.4f}")
        print(f"  Permutation test (slope≠0): p={p_perm_slope:.4f}")
        
        # ====================================================================
        # 3. First vs Last day comparison
        # ====================================================================
        first_day = days_with_data[0]
        last_day = days_with_data[-1]
        first_smi = day_layer_data[first_day][layer]
        last_smi = day_layer_data[last_day][layer]
        
        u_stat, p_fl = stats.mannwhitneyu(first_smi, last_smi, alternative='two-sided')
        delta, magnitude = cliffs_delta(last_smi, first_smi)
        change = medians_arr[-1] - medians_arr[0]
        
        print(f"\nFirst vs Last Day:")
        print(f"  Day {first_day} vs Day {last_day}: Δ={change:+.3f}, p={p_fl:.4f}")
        print(f"  Cliff's Delta: δ={delta:.3f} ({magnitude})")
        
        if p_fl < 0.05 and change > 0:
            print(f"  ✓ Significant INCREASE in {layer}")
        elif p_fl < 0.05 and change < 0:
            print(f"  ✗ Significant DECREASE in {layer}")
        else:
            print(f"  → No significant change")
        
        # Store results
        results['by_layer'][layer] = {
            'days': days_with_data,
            'medians': medians,
            'n_cells': n_cells,
            'spearman': {'rho': rho, 'p': p_spearman, 'ci_95': rho_ci},
            'linreg': {'slope': slope, 'r2': r_val**2, 'p': p_linreg},
            'permutation_slope': {'p': p_perm_slope, 'observed_slope': obs_slope},
            'first_vs_last': {'delta': change, 'p': p_fl, 'cliffs_delta': delta, 
                              'effect_magnitude': magnitude}
        }
    
    return results


# ============================================================================
# ANALYSIS 3: COMPARE LAYER DEVELOPMENT RATES
# ============================================================================

def compare_layer_development_rates(all_data, temporal_results):
    """
    Question 2b: Do layers develop at different rates?
    
    Statistical tests:
    - Permutation tests comparing slopes between layers
    - Kruskal-Wallis on delta SMI (early vs late)
    """
    print(f"\n{'='*70}")
    print("ANALYSIS 3: COMPARING LAYER DEVELOPMENT RATES")
    print(f"{'='*70}")
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    
    # Define early and late periods
    early_days = [1, 2, 3]
    late_days_min = 4
    
    results = {'slope_comparisons': {}, 'delta_smi': {}}
    
    # ====================================================================
    # 1. Pairwise slope comparisons (permutation tests)
    # ====================================================================
    print("\nPairwise Slope Comparisons (Permutation Tests):")
    print("-" * 70)
    
    slope_comparisons = {}
    for l1, l2 in combinations(layer_order, 2):
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
        
        slope_comparisons[f'{l1}_vs_{l2}'] = {
            'slope1': slope1,
            'slope2': slope2,
            'diff': obs_diff,
            'p_permutation': p_perm
        }
        
        sig = '***' if p_perm < 0.001 else '**' if p_perm < 0.01 else '*' if p_perm < 0.05 else ''
        print(f"  {l1} vs {l2}:")
        print(f"    Slopes: {slope1:.4f} vs {slope2:.4f}, diff={obs_diff:+.4f}")
        print(f"    Permutation test: p={p_perm:.4f} {sig}")
    
    results['slope_comparisons'] = slope_comparisons
    
    # ====================================================================
    # 2. Delta SMI analysis (Early vs Late)
    # ====================================================================
    print(f"\n{'='*70}")
    print("Delta SMI Analysis (Early Days 1-3 vs Late Days 4+)")
    print(f"{'='*70}")
    
    # Collect delta SMI per animal per layer
    delta_smi_by_layer = {layer: [] for layer in layer_order}
    
    for animal_id, animal_data in all_data.items():
        available_days = sorted(animal_data.keys())
        early_available = [d for d in available_days if d in early_days]
        late_available = [d for d in available_days if d >= late_days_min]
        
        if len(early_available) == 0 or len(late_available) == 0:
            continue
        
        for layer in layer_order:
            # Collect early SMI
            early_smi = []
            for day in early_available:
                if layer in animal_data[day]['layers']:
                    early_smi.extend(animal_data[day]['layers'][layer]['SMI'])
            
            # Collect late SMI
            late_smi = []
            for day in late_available:
                if layer in animal_data[day]['layers']:
                    late_smi.extend(animal_data[day]['layers'][layer]['SMI'])
            
            if len(early_smi) > 0 and len(late_smi) > 0:
                delta = np.median(late_smi) - np.median(early_smi)
                delta_smi_by_layer[layer].append(delta)
    
    # Print summary
    print("\nDelta SMI per layer (Late - Early):")
    for layer in layer_order:
        if len(delta_smi_by_layer[layer]) > 0:
            median_delta = np.median(delta_smi_by_layer[layer])
            n_animals = len(delta_smi_by_layer[layer])
            print(f"  {layer}: median Δ={median_delta:+.3f} (n={n_animals} animals)")
    
    # Kruskal-Wallis test
    valid_layers_delta = [l for l in layer_order if len(delta_smi_by_layer[l]) > 1]
    if len(valid_layers_delta) >= 2:
        groups_delta = [delta_smi_by_layer[l] for l in valid_layers_delta]
        h_stat, kw_p = stats.kruskal(*groups_delta)
        
        print(f"\nKruskal-Wallis (delta SMI across layers): H={h_stat:.3f}, p={kw_p:.4f}")
        
        results['delta_smi']['kruskal_wallis'] = {'H': h_stat, 'p': kw_p}
        results['delta_smi']['by_layer'] = delta_smi_by_layer
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_across_animals_visualizations(all_data, day1_results, temporal_results, 
                                         development_results, save_path=None):
    """
    Create comprehensive visualizations (2×3 grid, Option A).
    
    Panels:
    1. Line plot: SMI over time by layer (SEM + individual animals)
    2. Heatmap: Days × Layers (pooled median)
    3. Day 1: Violin plot comparing layers
    4. Early vs Late: Box plots by layer
    5. Superficial vs Deep: Trajectory comparison
    6. Slope comparison: Development rates between layers
    """
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    fig = plt.figure(figsize=(22, 14))
    # =========================================================================
    # Panel 1: Line plot with SEM (SIMPLIFIED - no individual trajectories)
    # =========================================================================
    ax1 = fig.add_subplot(2, 3, 1)

    # Organize data for SEM calculation (pooled cells)
    all_days = sorted(set(d for a in all_data.values() for d in a.keys()))

    for layer in layer_order:
        # Collect data per day
        days_plot = []
        medians_plot = []
        sems_plot = []
        
        for day in all_days:
            all_smi_this_day = []
            
            for animal_id, animal_data in all_data.items():
                if day in animal_data and layer in animal_data[day]['layers']:
                    smi_vals = animal_data[day]['layers'][layer]['SMI']
                    if len(smi_vals) > 0:
                        all_smi_this_day.extend(smi_vals)
            
            if len(all_smi_this_day) > 0:
                days_plot.append(day)
                medians_plot.append(np.median(all_smi_this_day))
                sems_plot.append(stats.sem(all_smi_this_day))
        
        days_plot = np.array(days_plot)
        medians_plot = np.array(medians_plot)
        sems_plot = np.array(sems_plot)
        
        # Plot population line with SEM
        ax1.plot(days_plot, medians_plot, 'o-', color=layer_colors[layer], linewidth=3,
                markersize=10, label=layer, zorder=3, markeredgecolor='white', 
                markeredgewidth=1.5)
        ax1.fill_between(days_plot, medians_plot - sems_plot, medians_plot + sems_plot,
                        color=layer_colors[layer], alpha=0.3, zorder=2, 
                        edgecolor=layer_colors[layer], linewidth=1.5)

    ax1.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Median SMI', fontsize=12, fontweight='bold')
    ax1.set_title('SMI Development Over Time\n(Shaded area = SEM from pooled cells)', 
                fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, framealpha=0.95, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, None])  # Start y-axis at 0
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # =========================================================================
    # Panel 2: Heatmap - Days × Layers WITH ANIMAL COUNTS
    # =========================================================================
    ax2 = fig.add_subplot(2, 3, 2)

    heatmap = np.full((len(layer_order), len(all_days)), np.nan)
    n_cells_matrix = np.zeros((len(layer_order), len(all_days)), dtype=int)
    n_animals_matrix = np.zeros((len(layer_order), len(all_days)), dtype=int)  # NEW

    for j, day in enumerate(all_days):
        for i, layer in enumerate(layer_order):
            all_smi = []
            animal_count = 0  # NEW
            
            for animal_data in all_data.values():
                if day in animal_data and layer in animal_data[day]['layers']:
                    smi_vals = animal_data[day]['layers'][layer]['SMI']
                    if len(smi_vals) > 0:
                        all_smi.extend(smi_vals)
                        animal_count += 1  # NEW
            
            if len(all_smi) > 0:
                heatmap[i, j] = np.median(all_smi)
                n_cells_matrix[i, j] = len(all_smi)
                n_animals_matrix[i, j] = animal_count  # NEW

    im = ax2.imshow(heatmap, cmap='YlOrRd', aspect='auto', vmin=0, vmax=np.nanmax(heatmap))
    ax2.set_xticks(range(len(all_days)))
    ax2.set_xticklabels([f'D{d}' for d in all_days])
    ax2.set_yticks(range(len(layer_order)))
    ax2.set_yticklabels(layer_order)

    # Annotate with median values, cell counts, AND animal counts
    for i in range(len(layer_order)):
        for j in range(len(all_days)):
            if not np.isnan(heatmap[i, j]):
                text_color = 'white' if heatmap[i, j] > np.nanmax(heatmap) * 0.6 else 'black'
                # UPDATED annotation format
                ax2.text(j, i, f'{heatmap[i, j]:.2f}\nn={n_cells_matrix[i, j]}\n({n_animals_matrix[i, j]} mice)', 
                        ha='center', va='center', fontsize=7.5, color=text_color,
                        weight='normal')

    ax2.set_title('Pooled Median SMI\n(cells from all animals)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Median SMI', shrink=0.8)
    
    # =========================================================================
    # Panel 3: Day 1 layer comparison - Violin plot (FIXED Y-AXIS)
    # =========================================================================
    ax3 = fig.add_subplot(2, 3, 3)

    pooled = day1_results['pooled_smi']
    plot_data = []
    plot_labels = []
    plot_colors = []
    positions = []

    for idx, layer in enumerate(layer_order):
        if len(pooled[layer]) > 0:
            plot_data.append(pooled[layer])
            plot_labels.append(f'{layer}\n(n={len(pooled[layer])})')
            plot_colors.append(layer_colors[layer])
            positions.append(idx)

    if plot_data:
        parts = ax3.violinplot(plot_data, positions=positions, widths=0.7,
                               showmeans=False, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plot_colors[i])
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Style median lines
        parts['cmedians'].set_edgecolor('black')
        parts['cmedians'].set_linewidth(2)
        
        # Add individual points (downsampled if too many)
        for i, data in enumerate(plot_data):
            data = np.array(data)  # FIXED: Ensure it's a numpy array
            
            if len(data) > 500:
                # Downsample for visualization
                sample_indices = np.random.choice(len(data), size=500, replace=False)
                y = data[sample_indices]
            else:
                y = data
            
            x = np.random.normal(positions[i], 0.04, size=len(y))
            ax3.scatter(x, y, alpha=0.2, s=1.5, color=plot_colors[i])
        
        ax3.set_xticks(positions)
        ax3.set_xticklabels(plot_labels)
        ax3.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax3.set_ylabel('SMI', fontsize=12, fontweight='bold')
        ax3.set_title('Day 1: Layer Comparison\n(pooled across all animals)', 
                     fontsize=13, fontweight='bold')
        ax3.set_ylim([-0.1, 1.0])  # FIXED: Focus on relevant range
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

    # =========================================================================
    # Panel 4: Early (1-3) vs Late (4+) comparison by layer
    # =========================================================================
    ax4 = fig.add_subplot(2, 3, 4)
    
    early_days = [1, 2, 3]
    late_days_min = 4
    
    early_data = {layer: [] for layer in layer_order}
    late_data = {layer: [] for layer in layer_order}
    
    for animal_data in all_data.values():
        for day, session in animal_data.items():
            for layer in layer_order:
                if layer in session['layers'] and len(session['layers'][layer]['SMI']) > 0:
                    smi_vals = session['layers'][layer]['SMI']
                    if day in early_days:
                        early_data[layer].extend(smi_vals)
                    elif day >= late_days_min:
                        late_data[layer].extend(smi_vals)
    
    x_pos = np.arange(len(layer_order))
    width = 0.35
    
    early_medians = [np.median(early_data[l]) if len(early_data[l]) > 0 else 0 for l in layer_order]
    late_medians = [np.median(late_data[l]) if len(late_data[l]) > 0 else 0 for l in layer_order]
    
    early_sems = [stats.sem(early_data[l]) if len(early_data[l]) > 1 else 0 for l in layer_order]
    late_sems = [stats.sem(late_data[l]) if len(late_data[l]) > 1 else 0 for l in layer_order]
    
    bars1 = ax4.bar(x_pos - width/2, early_medians, width, label='Early (D1-3)', 
                   yerr=early_sems, capsize=5, alpha=0.7, color='steelblue')
    bars2 = ax4.bar(x_pos + width/2, late_medians, width, label='Late (D4+)', 
                   yerr=late_sems, capsize=5, alpha=0.7, color='coral')
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(layer_order)
    ax4.set_ylabel('Median SMI', fontsize=12, fontweight='bold')
    ax4.set_title('Early vs Late Development by Layer', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Panel 5: Superficial vs Deep trajectory
    # =========================================================================
    ax5 = fig.add_subplot(2, 3, 5)
    
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
    sup_medians = [np.median(sup_by_day[d]) for d in days_common]
    deep_medians = [np.median(deep_by_day[d]) for d in days_common]
    sup_sems = [stats.sem(sup_by_day[d]) for d in days_common]
    deep_sems = [stats.sem(deep_by_day[d]) for d in days_common]
    
    ax5.plot(days_common, sup_medians, 'o-', color='#1E88E5', linewidth=3,
            markersize=12, label='Superficial (L2/3 + L4)')
    ax5.fill_between(days_common, 
                     np.array(sup_medians) - np.array(sup_sems),
                     np.array(sup_medians) + np.array(sup_sems),
                     color='#1E88E5', alpha=0.2)
    
    ax5.plot(days_common, deep_medians, 's-', color='#E53935', linewidth=3,
            markersize=12, label='Deep (L5 + L6)')
    ax5.fill_between(days_common,
                     np.array(deep_medians) - np.array(deep_sems),
                     np.array(deep_medians) + np.array(deep_sems),
                     color='#E53935', alpha=0.2)
    
    ax5.fill_between(days_common, sup_medians, deep_medians, alpha=0.1, color='gray')
    
    ax5.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Median SMI', fontsize=12, fontweight='bold')
    ax5.set_title('Superficial vs Deep Layers', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10, framealpha=0.9)
    ax5.grid(True, alpha=0.3)
    # =========================================================================
    # Panel 6: Slope comparison with BOOTSTRAP CI (OPTIONAL ENHANCEMENT)
    # =========================================================================
    ax6 = fig.add_subplot(2, 3, 6)

    slopes = []
    slope_cis = []  # NEW
    slope_labels = []
    slope_colors = []

    for layer in layer_order:
        if layer in temporal_results['by_layer']:
            slope = temporal_results['by_layer'][layer]['linreg']['slope']
            slopes.append(slope)
            slope_labels.append(layer)
            slope_colors.append(layer_colors[layer])
            
            # Bootstrap CI for slope (NEW)
            days = np.array(temporal_results['by_layer'][layer]['days'])
            medians = np.array(temporal_results['by_layer'][layer]['medians'])
            
            boot_slopes = []
            for _ in range(5000):
                indices = np.random.choice(len(days), size=len(days), replace=True)
                boot_slope, _, _, _, _ = stats.linregress(days[indices], medians[indices])
                boot_slopes.append(boot_slope)
            
            slope_ci_lower = np.percentile(boot_slopes, 2.5)
            slope_ci_upper = np.percentile(boot_slopes, 97.5)
            slope_cis.append((slope_ci_lower, slope_ci_upper))

    x_pos = np.arange(len(slopes))

    # Calculate error bar values
    yerr_lower = [slopes[i] - slope_cis[i][0] for i in range(len(slopes))]
    yerr_upper = [slope_cis[i][1] - slopes[i] for i in range(len(slopes))]
    yerr = [yerr_lower, yerr_upper]

    bars = ax6.bar(x_pos, slopes, color=slope_colors, alpha=0.7, edgecolor='black', 
                linewidth=1.5, yerr=yerr, capsize=5, error_kw={'linewidth': 2})

    # Add significance stars
    for i, layer in enumerate(slope_labels):
        p_val = temporal_results['by_layer'][layer]['permutation_slope']['p']
        if p_val < 0.001:
            sig_text = '***'
        elif p_val < 0.01:
            sig_text = '**'
        elif p_val < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        
        y_pos = slopes[i] + yerr_upper[i] + 0.002
        ax6.text(i, y_pos, sig_text, ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    ax6.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(slope_labels)
    ax6.set_ylabel('Slope (ΔSMI/day)', fontsize=12, fontweight='bold')
    ax6.set_title('Development Rate by Layer\n(Error bars = 95% CI, * p<0.05, ** p<0.01, *** p<0.001)', 
                fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    
    # Main title
    fig.suptitle('Across-Animals SMI Analysis', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        fig_path = os.path.join(save_path, 'across_animals_smi_analysis.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {fig_path}")
    
    return fig


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def create_summary_table(all_data, temporal_results, save_path=None):
    """Create and save summary statistics table."""
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    rows = []
    
    for animal_id in sorted(all_data.keys()):
        for day in sorted(all_data[animal_id].keys()):
            session = all_data[animal_id][day]
            row = {'Animal': animal_id, 'Day': day}
            
            for layer in layer_order:
                if layer in session['layers']:
                    ld = session['layers'][layer]
                    row[f'{layer}_n'] = ld['n_valid']
                    row[f'{layer}_median'] = ld['median_smi']
                else:
                    row[f'{layer}_n'] = 0
                    row[f'{layer}_median'] = np.nan
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Add slope information
    slope_rows = []
    for layer in layer_order:
        if layer in temporal_results['by_layer']:
            tr = temporal_results['by_layer'][layer]
            slope_rows.append({
                'Layer': layer,
                'Slope (ΔSMI/day)': tr['linreg']['slope'],
                'Spearman ρ': tr['spearman']['rho'],
                'Spearman p': tr['spearman']['p'],
                'Permutation p (slope)': tr['permutation_slope']['p']
            })
    
    df_slopes = pd.DataFrame(slope_rows)
    
    if save_path:
        csv_path = os.path.join(save_path, 'smi_summary_table.csv')
        df.to_csv(csv_path, index=False)
        
        slope_path = os.path.join(save_path, 'smi_slopes_table.csv')
        df_slopes.to_csv(slope_path, index=False)
        
        print(f"\n✓ Saved summary table: {csv_path}")
        print(f"✓ Saved slopes table: {slope_path}")
    
    return df, df_slopes


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def run_across_animals_analysis(parent_dir, save_path=None):
    """Complete workflow for across-animals SMI analysis."""
    print(f"\n{'='*80}")
    print("ACROSS-ANIMALS SMI ANALYSIS")
    print(f"{'='*80}")
    
    # Load all data
    all_data = load_all_animals_smi(parent_dir)
    
    if len(all_data) == 0:
        print("No data loaded. Exiting.")
        return None
    
    # Create save directory
    if save_path is None:
        save_path = os.path.join(parent_dir, 'across_animals_smi_analysis')
    os.makedirs(save_path, exist_ok=True)
    
    # Analysis 1: Day 1 layer differences
    day1_results = analyze_day1_layer_differences_pooled(all_data)
    
    # Analysis 2: Temporal progression
    temporal_results = analyze_temporal_progression_pooled(all_data)
    
    # Analysis 3: Compare development rates
    development_results = compare_layer_development_rates(all_data, temporal_results)
    
    # Create visualizations
    fig = create_across_animals_visualizations(all_data, day1_results, temporal_results,
                                               development_results, save_path)
    
    # Create summary tables
    summary_df, slopes_df = create_summary_table(all_data, temporal_results, save_path)
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY TABLE (First 10 rows)")
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
        'summary_table': summary_df,
        'slopes_table': slopes_df,
        'figure': fig
    }
    
    print(f"\n{'='*80}")
    print("ACROSS-ANIMALS ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {save_path}")
    print(f"  - across_animals_smi_analysis.png")
    print(f"  - smi_summary_table.csv")
    print(f"  - smi_slopes_table.csv")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Configure paths
    parent_dir = r"D:\V1_SpatialModulation\2p\V1_prism"
    save_dir = r"D:\V1_SpatialModulation\2p\V1_prism\across_animals_smi_analysis"
    
    results = run_across_animals_analysis(parent_dir, save_dir)
    
    plt.show()
