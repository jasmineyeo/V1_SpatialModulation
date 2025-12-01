"""
SMI_CompareAcrossAnimals_pt1.py (IMPROVED) - PART 1
Data Loading and Statistical Analyses

IMPROVEMENTS:
1. Early vs Late comparison (pooled across animals)
2. Gap closure analysis (pooled)
3. Individual animal trajectory tracking
4. Effect size reporting (Cohen's d, rank-biserial)
5. Proportion of modulated cells analysis

Run Part 1 first, then Part 2 for visualizations.

JSY, 2025
"""
import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import h5py
from glob import glob
from collections import defaultdict
import pandas as pd


# =============================================================================
# DATA LOADING
# =============================================================================

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
                            'SMI': smi_vals,
                            'prop_modulated': prop_mod
                        }
                
                all_data[animal_id][day] = session_data
                
        except Exception as e:
            print(f"    ERROR: {e}")
    
    animals = list(all_data.keys())
    print(f"\nLoaded {len(animals)} animals:")
    for animal in sorted(animals):
        days = sorted(all_data[animal].keys())
        print(f"  {animal}: Days {days}")
    
    return dict(all_data)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compute_rank_biserial(u_stat, n1, n2):
    """Compute rank-biserial correlation from Mann-Whitney U."""
    return 1 - (2 * u_stat) / (n1 * n2)


# =============================================================================
# ANALYSIS 1: DAY 1 LAYER DIFFERENCES (POOLED)
# =============================================================================

def analyze_day1_layer_differences_pooled(all_data):
    """Question 1: On Day 1, do deeper layers have higher SMI?"""
    
    print(f"\n{'='*70}")
    print("ANALYSIS 1: DAY 1 LAYER DIFFERENCES (POOLED)")
    print(f"{'='*70}")
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    
    pooled_smi = {layer: [] for layer in layer_order}
    animal_medians = {layer: [] for layer in layer_order}
    animal_ids_per_layer = {layer: [] for layer in layer_order}
    
    for animal_id, animal_data in all_data.items():
        day1 = min(animal_data.keys())
        if day1 > 2:
            continue
        
        session = animal_data[day1]
        for layer in layer_order:
            if layer in session['layers'] and len(session['layers'][layer]['SMI']) > 0:
                smi_vals = session['layers'][layer]['SMI']
                pooled_smi[layer].extend(smi_vals)
                animal_medians[layer].append(session['layers'][layer]['median_smi'])
                animal_ids_per_layer[layer].append(animal_id)
    
    print("\nPooled SMI values (Day 1):")
    for layer in layer_order:
        if len(pooled_smi[layer]) > 0:
            med = np.median(pooled_smi[layer])
            mad = stats.median_abs_deviation(pooled_smi[layer])
            n_cells = len(pooled_smi[layer])
            n_animals = len(animal_medians[layer])
            print(f"  {layer}: n={n_cells} cells from {n_animals} animals, median={med:.3f}±{mad:.3f}")
    
    valid_layers = [l for l in layer_order if len(pooled_smi[l]) > 0]
    results = {'pooled_smi': pooled_smi, 'animal_medians': animal_medians}
    
    if len(valid_layers) >= 2:
        # Kruskal-Wallis
        groups = [pooled_smi[l] for l in valid_layers]
        h_stat, kw_p = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis: H={h_stat:.3f}, p={kw_p:.2e}")
        results['kruskal_wallis'] = {'H': h_stat, 'p': kw_p}
        
        # Pairwise with effect sizes
        print("\nPairwise comparisons:")
        pairwise = {}
        for i, l1 in enumerate(valid_layers):
            for l2 in valid_layers[i+1:]:
                u_stat, p_val = stats.mannwhitneyu(pooled_smi[l1], pooled_smi[l2], alternative='two-sided')
                d = compute_cohens_d(pooled_smi[l1], pooled_smi[l2])
                r = compute_rank_biserial(u_stat, len(pooled_smi[l1]), len(pooled_smi[l2]))
                
                pairwise[f'{l1}_vs_{l2}'] = {'U': u_stat, 'p': p_val, 'cohens_d': d, 'rank_biserial': r}
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                print(f"  {l1} vs {l2}: p={p_val:.4f} {sig}, d={d:.3f}, r={r:.3f}")
        
        results['pairwise'] = pairwise
        
        # Key: Superficial vs Deep
        if 'L2/3' in valid_layers and ('L5' in valid_layers or 'L6' in valid_layers):
            sup_smi = pooled_smi['L2/3']
            deep_smi = []
            for l in ['L5', 'L6']:
                if l in valid_layers:
                    deep_smi.extend(pooled_smi[l])
            
            u_stat, p_val = stats.mannwhitneyu(sup_smi, deep_smi, alternative='less')
            d = compute_cohens_d(deep_smi, sup_smi)
            
            print(f"\n  KEY TEST - Deep > Superficial (one-sided):")
            print(f"    L2/3 median: {np.median(sup_smi):.3f}, Deep median: {np.median(deep_smi):.3f}")
            print(f"    p={p_val:.4f}, Cohen's d={d:.3f}")
            
            results['sup_vs_deep'] = {'p': p_val, 'cohens_d': d, 
                                       'sup_median': np.median(sup_smi), 
                                       'deep_median': np.median(deep_smi)}
    
    return results


# =============================================================================
# ANALYSIS 2: EARLY VS LATE (POOLED)
# =============================================================================

def analyze_early_vs_late_pooled(all_data):
    """Compare early (Days 1-2) vs late (Days 6-7) - pooled across animals."""
    
    print(f"\n{'='*70}")
    print("ANALYSIS 2: EARLY vs LATE (POOLED ACROSS ANIMALS)")
    print(f"{'='*70}")
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    results = {}
    
    for layer in layer_order:
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
            print(f"\n  {layer}: Insufficient data")
            continue
        
        early_smi = np.array(early_smi)
        late_smi = np.array(late_smi)
        
        u_stat, p_val = stats.mannwhitneyu(early_smi, late_smi, alternative='less')
        d = compute_cohens_d(late_smi, early_smi)
        r = compute_rank_biserial(u_stat, len(early_smi), len(late_smi))
        
        early_med = np.median(early_smi)
        late_med = np.median(late_smi)
        delta = late_med - early_med
        
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        
        print(f"\n  {layer}:")
        print(f"    Early (n={len(early_smi)} from {len(set(early_animals))} animals): median={early_med:.3f}")
        print(f"    Late (n={len(late_smi)} from {len(set(late_animals))} animals): median={late_med:.3f}")
        print(f"    Δ={delta:+.3f}, p={p_val:.4f} {sig}, d={d:.3f}")
        
        results[layer] = {
            'early_median': early_med, 'late_median': late_med,
            'delta': delta, 'p_value': p_val,
            'cohens_d': d, 'rank_biserial': r,
            'early_n': len(early_smi), 'late_n': len(late_smi),
            'early_n_animals': len(set(early_animals)),
            'late_n_animals': len(set(late_animals))
        }
    
    return results


# =============================================================================
# ANALYSIS 3: TEMPORAL PROGRESSION (POOLED)
# =============================================================================

def analyze_temporal_progression_pooled(all_data):
    """Track SMI development over days - pooled across animals."""
    
    print(f"\n{'='*70}")
    print("ANALYSIS 3: TEMPORAL PROGRESSION (POOLED)")
    print(f"{'='*70}")
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    
    day_layer_data = defaultdict(lambda: {l: [] for l in layer_order})
    
    all_days = set()
    for animal_data in all_data.values():
        for day, session in animal_data.items():
            all_days.add(day)
            for layer in layer_order:
                if layer in session['layers'] and len(session['layers'][layer]['SMI']) > 0:
                    day_layer_data[day][layer].extend(session['layers'][layer]['SMI'])
    
    days_sorted = sorted(all_days)
    results = {'by_layer': {}, 'days': days_sorted}
    
    for layer in layer_order:
        print(f"\n--- {layer} ---")
        
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
        
        if len(days_with_data) < 3:
            continue
        
        days_arr = np.array(days_with_data)
        medians_arr = np.array(medians)
        
        rho, p_spearman = stats.spearmanr(days_arr, medians_arr)
        slope, intercept, r_val, p_linreg, _ = stats.linregress(days_arr, medians_arr)
        
        print(f"\n  Spearman: ρ={rho:.3f}, p={p_spearman:.4f}")
        print(f"  Linear: slope={slope:.4f}/day, R²={r_val**2:.3f}")
        
        results['by_layer'][layer] = {
            'days': days_with_data, 'medians': medians, 'n_cells': n_cells,
            'spearman': {'rho': rho, 'p': p_spearman},
            'linreg': {'slope': slope, 'r2': r_val**2, 'p': p_linreg}
        }
    
    return results


# =============================================================================
# ANALYSIS 4: GAP CLOSURE (POOLED)
# =============================================================================

def analyze_gap_closure_pooled(all_data):
    """Track Deep - Superficial gap over time (pooled)."""
    
    print(f"\n{'='*70}")
    print("ANALYSIS 4: GAP CLOSURE (POOLED)")
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
        return None
    
    gaps = np.array(gaps)
    gap_days = np.array(gap_days)
    
    rho, p_val = stats.spearmanr(gap_days, gaps)
    slope, intercept, r_val, p_linreg, _ = stats.linregress(gap_days, gaps)
    
    print(f"\n  Gap trend: ρ={rho:.3f}, p={p_val:.4f}")
    print(f"  Linear: slope={slope:.4f}/day")
    print(f"  Initial gap: {gaps[0]:.3f}, Final gap: {gaps[-1]:.3f}")
    
    if rho < 0 and p_val < 0.05:
        print("  ✓ Gap SIGNIFICANTLY DECREASES (superficial catching up)")
    elif rho < 0:
        print("  → Trend toward gap closure (not significant)")
    
    return {
        'days': gap_days.tolist(), 'gaps': gaps.tolist(),
        'sup_medians': sup_medians, 'deep_medians': deep_medians,
        'spearman_rho': rho, 'spearman_p': p_val,
        'slope': slope, 'initial_gap': gaps[0], 'final_gap': gaps[-1]
    }


# =============================================================================
# ANALYSIS 5: INDIVIDUAL ANIMAL TRAJECTORIES
# =============================================================================

def analyze_individual_trajectories(all_data):
    """Track each animal's SMI trajectory for consistency check."""
    
    print(f"\n{'='*70}")
    print("ANALYSIS 5: INDIVIDUAL ANIMAL TRAJECTORIES")
    print(f"{'='*70}")
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    
    trajectories = {layer: {} for layer in layer_order}
    
    for animal_id in sorted(all_data.keys()):
        animal_data = all_data[animal_id]
        
        for layer in layer_order:
            days = []
            medians = []
            
            for day in sorted(animal_data.keys()):
                if layer in animal_data[day]['layers']:
                    ld = animal_data[day]['layers'][layer]
                    if ld['n_valid'] > 0:
                        days.append(day)
                        medians.append(ld['median_smi'])
            
            if len(days) >= 2:
                # Calculate slope for this animal
                if len(days) >= 3:
                    rho, _ = stats.spearmanr(days, medians)
                else:
                    rho = np.nan
                
                trajectories[layer][animal_id] = {
                    'days': days, 'medians': medians,
                    'spearman_rho': rho,
                    'change': medians[-1] - medians[0] if len(medians) >= 2 else np.nan
                }
    
    # Summary
    print("\nTrajectory summary (first → last day change):")
    for layer in layer_order:
        if trajectories[layer]:
            changes = [t['change'] for t in trajectories[layer].values() if not np.isnan(t['change'])]
            if changes:
                mean_change = np.mean(changes)
                n_pos = sum(c > 0 for c in changes)
                print(f"  {layer}: {n_pos}/{len(changes)} animals show increase, mean Δ={mean_change:+.3f}")
    
    return trajectories


# =============================================================================
# CREATE SUMMARY TABLE
# =============================================================================

def create_summary_table(all_data, save_path=None):
    """Create comprehensive summary table."""
    
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
                    row[f'{layer}_median'] = round(ld['median_smi'], 3)
                    row[f'{layer}_prop_mod'] = round(ld['prop_modulated'], 3) if not np.isnan(ld['prop_modulated']) else np.nan
                else:
                    row[f'{layer}_n'] = 0
                    row[f'{layer}_median'] = np.nan
                    row[f'{layer}_prop_mod'] = np.nan
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if save_path:
        csv_path = os.path.join(save_path, 'smi_summary_table.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved: {csv_path}")
    
    return df


# =============================================================================
# MAIN ANALYSIS FUNCTION (Part 1)
# =============================================================================

def run_analyses(parent_dir, save_path=None):
    """Run all analyses (Part 1) - call this before visualization."""
    
    print(f"\n{'='*80}")
    print("ACROSS-ANIMALS SMI ANALYSIS - PART 1 (DATA & STATISTICS)")
    print(f"{'='*80}")
    
    # Load data
    all_data = load_all_animals_smi(parent_dir)
    
    if len(all_data) == 0:
        print("No data loaded.")
        return None
    
    # Create save directory
    if save_path is None:
        save_path = os.path.join(parent_dir, 'across_animals_smi_analysis')
    os.makedirs(save_path, exist_ok=True)
    
    # Run analyses
    day1_results = analyze_day1_layer_differences_pooled(all_data)
    early_late_results = analyze_early_vs_late_pooled(all_data)
    temporal_results = analyze_temporal_progression_pooled(all_data)
    gap_results = analyze_gap_closure_pooled(all_data)
    trajectory_results = analyze_individual_trajectories(all_data)
    summary_df = create_summary_table(all_data, save_path)
    
    results = {
        'all_data': all_data,
        'day1_layer_differences': day1_results,
        'early_vs_late': early_late_results,
        'temporal_progression': temporal_results,
        'gap_closure': gap_results,
        'individual_trajectories': trajectory_results,
        'summary_table': summary_df,
        'save_path': save_path
    }
    
    print(f"\n{'='*70}")
    print("PART 1 COMPLETE - Run Part 2 for visualizations")
    print(f"{'='*70}")
    
    return results


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    parent_dir = r"D:\V1_SpatialModulation\2p\V1_prism"
    save_dir = r"D:\V1_SpatialModulation\2p\V1_prism\across_animals_smi_analysis"
    
    # Run Part 1
    results = run_analyses(parent_dir, save_dir)
    
    # Save results for Part 2
    if results:
        import pickle
        with open(os.path.join(save_dir, 'analysis_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        print(f"\n✓ Results saved for Part 2")