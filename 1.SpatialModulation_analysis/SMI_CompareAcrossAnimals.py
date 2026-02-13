"""
SMI_CompareAcrossAnimals.py
Cross-animal SMI comparison: data loading, statistical analyses, and visualization.

Analyses:
1. Day 1 layer differences (pooled across animals)
2. Early vs Late comparison (pooled)
3. Temporal progression (pooled)
4. Gap closure analysis (pooled)
5. Individual animal trajectory tracking
6. Summary table + 3x3 publication figure

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
# CONFIGURATION
# =============================================================================

LAYER_ORDER = ['L2/3', 'L4', 'L5', 'L6']
LAYER_COLORS = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}


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

    pooled_smi = {layer: [] for layer in LAYER_ORDER}
    animal_medians = {layer: [] for layer in LAYER_ORDER}
    animal_ids_per_layer = {layer: [] for layer in LAYER_ORDER}

    for animal_id, animal_data in all_data.items():
        day1 = min(animal_data.keys())
        if day1 > 2:
            continue

        session = animal_data[day1]
        for layer in LAYER_ORDER:
            if layer in session['layers'] and len(session['layers'][layer]['SMI']) > 0:
                smi_vals = session['layers'][layer]['SMI']
                pooled_smi[layer].extend(smi_vals)
                animal_medians[layer].append(session['layers'][layer]['median_smi'])
                animal_ids_per_layer[layer].append(animal_id)

    print("\nPooled SMI values (Day 1):")
    for layer in LAYER_ORDER:
        if len(pooled_smi[layer]) > 0:
            med = np.median(pooled_smi[layer])
            mad = stats.median_abs_deviation(pooled_smi[layer])
            n_cells = len(pooled_smi[layer])
            n_animals = len(animal_medians[layer])
            print(f"  {layer}: n={n_cells} cells from {n_animals} animals, median={med:.3f}±{mad:.3f}")

    valid_layers = [l for l in LAYER_ORDER if len(pooled_smi[l]) > 0]
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

    day_layer_data = defaultdict(lambda: {l: [] for l in LAYER_ORDER})

    all_days = set()
    for animal_data in all_data.values():
        for day, session in animal_data.items():
            all_days.add(day)
            for layer in LAYER_ORDER:
                if layer in session['layers'] and len(session['layers'][layer]['SMI']) > 0:
                    day_layer_data[day][layer].extend(session['layers'][layer]['SMI'])

    days_sorted = sorted(all_days)
    results = {'by_layer': {}, 'days': days_sorted}

    for layer in LAYER_ORDER:
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
        print("  Gap SIGNIFICANTLY DECREASES (superficial catching up)")
    elif rho < 0:
        print("  Trend toward gap closure (not significant)")

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
    print("\nTrajectory summary (first -> last day change):")
    for layer in LAYER_ORDER:
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

    rows = []
    for animal_id in sorted(all_data.keys()):
        for day in sorted(all_data[animal_id].keys()):
            session = all_data[animal_id][day]

            row = {'Animal': animal_id, 'Day': day}

            for layer in LAYER_ORDER:
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
        print(f"\nSaved: {csv_path}")

    return df


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_analyses(parent_dir, save_path=None):
    """Run all statistical analyses."""

    print(f"\n{'='*80}")
    print("ACROSS-ANIMALS SMI ANALYSIS - STATISTICS")
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

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_main_figure(results, save_path=None):
    """Create main 3x3 figure with key results."""

    all_data = results['all_data']
    day1_results = results['day1_layer_differences']
    early_late = results['early_vs_late']
    temporal = results['temporal_progression']
    gap = results['gap_closure']

    fig = plt.figure(figsize=(18, 16))

    # =========================================================================
    # Panel 1: Day 1 Violin Plot
    # =========================================================================
    ax1 = fig.add_subplot(3, 3, 1)

    pooled = day1_results['pooled_smi']
    plot_data, plot_labels, plot_colors = [], [], []

    for layer in LAYER_ORDER:
        if len(pooled[layer]) > 0:
            plot_data.append(pooled[layer])
            plot_labels.append(f'{layer}\n(n={len(pooled[layer])})')
            plot_colors.append(LAYER_COLORS[layer])

    if plot_data:
        parts = ax1.violinplot(plot_data, positions=range(len(plot_data)), widths=0.7,
                               showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plot_colors[i])
            pc.set_alpha(0.6)
        ax1.set_xticks(range(len(plot_labels)))
        ax1.set_xticklabels(plot_labels)

    # Add significance
    if 'kruskal_wallis' in day1_results:
        kw_p = day1_results['kruskal_wallis']['p']
        sig = '***' if kw_p < 0.001 else '**' if kw_p < 0.01 else '*' if kw_p < 0.05 else 'ns'
        ax1.text(0.95, 0.95, f'KW p={kw_p:.2e} {sig}', transform=ax1.transAxes,
                ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('SMI')
    ax1.set_title('Day 1: Layer Comparison (Pooled)', fontsize=12, fontweight='bold')
    ax1.set_ylim(-0.5, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Panel 2: Temporal Progression
    # =========================================================================
    ax2 = fig.add_subplot(3, 3, 2)

    for layer in LAYER_ORDER:
        if layer in temporal['by_layer']:
            tr = temporal['by_layer'][layer]
            ax2.plot(tr['days'], tr['medians'], 'o-', color=LAYER_COLORS[layer],
                    linewidth=2.5, markersize=10, label=layer)

    ax2.set_xlabel('Day')
    ax2.set_ylabel('Median SMI (pooled)')
    ax2.set_title('SMI Development Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # =========================================================================
    # Panel 3: Heatmap
    # =========================================================================
    ax3 = fig.add_subplot(3, 3, 3)

    all_days = sorted(set(d for a in all_data.values() for d in a.keys()))
    heatmap = np.full((len(LAYER_ORDER), len(all_days)), np.nan)

    for j, day in enumerate(all_days):
        for i, layer in enumerate(LAYER_ORDER):
            all_smi = []
            for animal_data in all_data.values():
                if day in animal_data and layer in animal_data[day]['layers']:
                    all_smi.extend(animal_data[day]['layers'][layer]['SMI'])
            if len(all_smi) > 0:
                heatmap[i, j] = np.median(all_smi)

    im = ax3.imshow(heatmap, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.7)
    ax3.set_xticks(range(len(all_days)))
    ax3.set_xticklabels([f'D{d}' for d in all_days])
    ax3.set_yticks(range(len(LAYER_ORDER)))
    ax3.set_yticklabels(LAYER_ORDER)

    for i in range(len(LAYER_ORDER)):
        for j in range(len(all_days)):
            if not np.isnan(heatmap[i, j]):
                ax3.text(j, i, f'{heatmap[i, j]:.2f}', ha='center', va='center', fontsize=8)

    ax3.set_title('Pooled Median SMI', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Median SMI', shrink=0.8)

    # =========================================================================
    # Panel 4: Early vs Late
    # =========================================================================
    ax4 = fig.add_subplot(3, 3, 4)

    if early_late:
        x = np.arange(len(LAYER_ORDER))
        width = 0.35

        early_vals = [early_late.get(l, {}).get('early_median', np.nan) for l in LAYER_ORDER]
        late_vals = [early_late.get(l, {}).get('late_median', np.nan) for l in LAYER_ORDER]

        ax4.bar(x - width/2, early_vals, width, label='Early (D1-2)', color='lightblue', edgecolor='black')
        ax4.bar(x + width/2, late_vals, width, label='Late (D6-7)', color='salmon', edgecolor='black')

        # Significance markers
        for i, layer in enumerate(LAYER_ORDER):
            if layer in early_late:
                p = early_late[layer]['p_value']
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                if sig:
                    max_val = max(early_vals[i], late_vals[i])
                    ax4.text(i, max_val + 0.03, sig, ha='center', fontsize=12, fontweight='bold')

        ax4.set_xticks(x)
        ax4.set_xticklabels(LAYER_ORDER)
        ax4.legend()

    ax4.set_ylabel('Median SMI')
    ax4.set_title('Early vs Late Development', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Panel 5: Superficial vs Deep
    # =========================================================================
    ax5 = fig.add_subplot(3, 3, 5)

    if gap:
        ax5.plot(gap['days'], gap['sup_medians'], 'o-', color='#1E88E5', linewidth=2.5,
                markersize=10, label='Superficial (L2/3+L4)')
        ax5.plot(gap['days'], gap['deep_medians'], 's-', color='#E53935', linewidth=2.5,
                markersize=10, label='Deep (L5+L6)')
        ax5.fill_between(gap['days'], gap['sup_medians'], gap['deep_medians'], alpha=0.2, color='gray')

        # Annotate gaps
        ax5.annotate(f"Gap={gap['initial_gap']:.2f}", xy=(gap['days'][0],
                    (gap['sup_medians'][0] + gap['deep_medians'][0])/2), fontsize=9)
        ax5.annotate(f"Gap={gap['final_gap']:.2f}", xy=(gap['days'][-1],
                    (gap['sup_medians'][-1] + gap['deep_medians'][-1])/2), fontsize=9, ha='right')

    ax5.set_xlabel('Day')
    ax5.set_ylabel('Median SMI')
    ax5.set_title('Superficial vs Deep Layers', fontsize=12, fontweight='bold')
    ax5.legend(loc='lower right')
    ax5.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 6: Gap Closure
    # =========================================================================
    ax6 = fig.add_subplot(3, 3, 6)

    if gap:
        ax6.plot(gap['days'], gap['gaps'], 'ko-', linewidth=2, markersize=10)
        ax6.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax6.fill_between(gap['days'], 0, gap['gaps'], alpha=0.3, color='purple')

        # Trend line
        z = np.polyfit(gap['days'], gap['gaps'], 1)
        p = np.poly1d(z)
        ax6.plot(gap['days'], p(gap['days']), 'r--', alpha=0.7, label=f"slope={gap['slope']:.3f}/day")

        ax6.text(0.95, 0.95, f"rho={gap['spearman_rho']:.2f}\np={gap['spearman_p']:.3f}",
                transform=ax6.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
        ax6.legend(loc='lower left')

    ax6.set_xlabel('Day')
    ax6.set_ylabel('Gap (Deep - Superficial)')
    ax6.set_title('Gap Closure Analysis', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 7: Individual Animal Trajectories (L2/3)
    # =========================================================================
    ax7 = fig.add_subplot(3, 3, 7)

    traj = results['individual_trajectories']
    if 'L2/3' in traj:
        for animal_id, t in traj['L2/3'].items():
            ax7.plot(t['days'], t['medians'], 'o-', alpha=0.5, linewidth=1.5, label=animal_id)

        # Add pooled trend
        if 'L2/3' in temporal['by_layer']:
            tr = temporal['by_layer']['L2/3']
            ax7.plot(tr['days'], tr['medians'], 'k-', linewidth=3, label='Pooled', zorder=10)

    ax7.set_xlabel('Day')
    ax7.set_ylabel('Median SMI')
    ax7.set_title('L2/3: Individual Trajectories', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=8, loc='lower right')
    ax7.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 8: Per-Animal Day 1 Comparison
    # =========================================================================
    ax8 = fig.add_subplot(3, 3, 8)

    animals = sorted(all_data.keys())
    x_pos = np.arange(len(animals))
    width = 0.2

    for i, layer in enumerate(LAYER_ORDER):
        medians = []
        for animal in animals:
            day1 = min(all_data[animal].keys())
            if layer in all_data[animal][day1]['layers']:
                medians.append(all_data[animal][day1]['layers'][layer]['median_smi'])
            else:
                medians.append(np.nan)

        offset = (i - 1.5) * width
        ax8.bar(x_pos + offset, medians, width, label=layer, color=LAYER_COLORS[layer], alpha=0.7)

    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(animals, rotation=45, ha='right')
    ax8.set_ylabel('Median SMI')
    ax8.set_title('Day 1: Per-Animal', fontsize=12, fontweight='bold')
    ax8.legend(loc='upper right', fontsize=8)
    ax8.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Panel 9: Statistical Summary
    # =========================================================================
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')

    n_animals = len(all_data)
    all_days = sorted(set(d for a in all_data.values() for d in a.keys()))

    summary = f"STATISTICAL SUMMARY\n{'='*40}\n\n"
    summary += f"Animals: {n_animals}\n"
    summary += f"Days: {min(all_days)} to {max(all_days)}\n\n"

    # Day 1
    summary += "DAY 1 LAYER DIFFERENCES:\n"
    if 'kruskal_wallis' in day1_results:
        kw_p = day1_results['kruskal_wallis']['p']
        sig = '***' if kw_p < 0.001 else '**' if kw_p < 0.01 else '*' if kw_p < 0.05 else ''
        summary += f"  Kruskal-Wallis: p={kw_p:.2e} {sig}\n"
    if 'sup_vs_deep' in day1_results:
        svd = day1_results['sup_vs_deep']
        summary += f"  Deep > Sup: p={svd['p']:.4f}, d={svd['cohens_d']:.2f}\n"

    # Early vs Late
    summary += "\nEARLY vs LATE:\n"
    for layer in LAYER_ORDER:
        if layer in early_late:
            el = early_late[layer]
            sig = '*' if el['p_value'] < 0.05 else ''
            summary += f"  {layer}: delta={el['delta']:+.3f}, p={el['p_value']:.3f}, d={el['cohens_d']:.2f} {sig}\n"

    # Gap closure
    if gap:
        summary += f"\nGAP CLOSURE:\n"
        summary += f"  Initial: {gap['initial_gap']:.3f}, Final: {gap['final_gap']:.3f}\n"
        summary += f"  Trend: rho={gap['spearman_rho']:.2f}, p={gap['spearman_p']:.3f}\n"

    ax9.text(0.05, 0.95, summary, transform=ax9.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Across-Animals SMI Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig_path = os.path.join(save_path, 'across_animals_smi_analysis.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {fig_path}")

    return fig


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    parent_dir = r"D:\V1_SpatialModulation\2p\V1_prism"
    save_dir = r"D:\V1_SpatialModulation\2p\V1_prism\across_animals_smi_analysis"

    # Run analyses
    results = run_analyses(parent_dir, save_dir)

    # Generate figures
    if results:
        fig = create_main_figure(results, save_dir)
        plt.show()

        print(f"\n{'='*70}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*70}")
