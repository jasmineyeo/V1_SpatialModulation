"""
SMICalculation_CompareSessionsWithinAnimal.py (IMPROVED)
Compare SMI development across sessions within a single animal.

IMPROVEMENTS:
1. Added early vs late comparison (Days 1-2 vs Days 6-7)
2. Added proportion of spatially modulated cells metric
3. Better statistical tests for small sample sizes
4. Cleaner visualizations with significance markers
5. Gap closure analysis (do superficial layers catch up?)

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


def extract_session_info(filepath):
    """Extract animal ID and session day from file path."""
    match_day = re.search(r'Day(\d+)', filepath, re.IGNORECASE)
    match_animal = re.search(r'(JSY\d+)', filepath)
    day = int(match_day.group(1)) if match_day else None
    animal = match_animal.group(1) if match_animal else None
    return animal, day


def load_animal_smi_data(animal_dir, smi_pattern="*_smi_results.h5"):
    """Load all SMI result files for a single animal."""
    
    print(f"\n{'='*70}")
    print(f"LOADING SMI DATA")
    print(f"{'='*70}")
    
    smi_files = glob(os.path.join(animal_dir, "**", smi_pattern), recursive=True)
    
    if len(smi_files) == 0:
        print(f"WARNING: No SMI result files found!")
        return {}, None
    
    print(f"Found {len(smi_files)} SMI result files\n")
    
    sessions_data = {}
    animal_id = None
    
    for smi_path in sorted(smi_files):
        animal, day = extract_session_info(smi_path)
        if animal_id is None:
            animal_id = animal
        if day is None:
            continue
        
        print(f"  Loading Day {day}: {os.path.basename(smi_path)}")
        
        try:
            with h5py.File(smi_path, 'r') as f:
                session_data = {
                    'session_id': f.attrs.get('session_id', f'Day{day}'),
                    'date': f.attrs.get('date', 'unknown'),
                    'day': day,
                    'global': {},
                    'layers': {}
                }
                
                if 'global_smi' in f:
                    g = f['global_smi']
                    session_data['global'] = {
                        'SMI_all': g['SMI_all_cells'][:],
                        'valid_mask': g['valid_cells_mask'][:],
                        'n_valid': g.attrs.get('n_valid_cells', 0),
                        'median_smi': g.attrs.get('median_smi', np.nan),
                    }
                
                if 'layer_smi' in f:
                    for layer_key in f['layer_smi'].keys():
                        lg = f['layer_smi'][layer_key]
                        layer_name = lg.attrs.get('original_name', layer_key.replace('_', '/'))
                        smi_vals = lg['SMI'][:] if 'SMI' in lg else np.array([])
                        
                        # Calculate proportion modulated (SMI > threshold)
                        prop_mod = np.mean(smi_vals > 0.1) if len(smi_vals) > 0 else np.nan
                        prop_strong = np.mean(smi_vals > 0.3) if len(smi_vals) > 0 else np.nan
                        
                        session_data['layers'][layer_name] = {
                            'n_valid': lg.attrs.get('n_cells_valid', len(smi_vals)),
                            'median_smi': lg.attrs.get('median_smi', np.nan),
                            'mean_smi': lg.attrs.get('mean_smi', np.nan),
                            'SMI': smi_vals,
                            'prop_modulated': prop_mod,
                            'prop_strong_mod': prop_strong
                        }
                
                sessions_data[day] = session_data
                
        except Exception as e:
            print(f"    ERROR: {e}")
    
    print(f"\nLoaded {len(sessions_data)} sessions for {animal_id}")
    return sessions_data, animal_id


def analyze_early_vs_late(sessions_data, animal_id):
    """
    NEW: Compare early days (1-2) vs late days (6-7) - more statistical power
    """
    
    print(f"\n{'='*70}")
    print("ANALYSIS: EARLY vs LATE COMPARISON")
    print(f"{'='*70}")
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    days = sorted(sessions_data.keys())
    
    # Define early and late periods
    early_days = [d for d in days if d <= 2]
    late_days = [d for d in days if d >= 6]
    
    if len(early_days) == 0 or len(late_days) == 0:
        print("  Insufficient early/late data")
        return None
    
    print(f"  Early days: {early_days}")
    print(f"  Late days: {late_days}")
    
    results = {}
    
    for layer in layer_order:
        # Pool early SMI values
        early_smi = []
        for day in early_days:
            if layer in sessions_data[day]['layers']:
                early_smi.extend(sessions_data[day]['layers'][layer]['SMI'])
        
        # Pool late SMI values
        late_smi = []
        for day in late_days:
            if layer in sessions_data[day]['layers']:
                late_smi.extend(sessions_data[day]['layers'][layer]['SMI'])
        
        if len(early_smi) < 5 or len(late_smi) < 5:
            print(f"\n  {layer}: Insufficient data (early={len(early_smi)}, late={len(late_smi)})")
            continue
        
        early_smi = np.array(early_smi)
        late_smi = np.array(late_smi)
        
        # Mann-Whitney U test
        u_stat, p_val = stats.mannwhitneyu(early_smi, late_smi, alternative='less')  # one-sided: late > early
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(early_smi), len(late_smi)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)
        
        early_med = np.median(early_smi)
        late_med = np.median(late_smi)
        delta = late_med - early_med
        
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        
        print(f"\n  {layer}:")
        print(f"    Early (n={len(early_smi)}): median={early_med:.3f}")
        print(f"    Late (n={len(late_smi)}): median={late_med:.3f}")
        print(f"    Δ = {delta:+.3f}, p={p_val:.4f} (one-sided) {sig}")
        print(f"    Effect size (r) = {effect_size:.3f}")
        
        results[layer] = {
            'early_median': early_med,
            'late_median': late_med,
            'delta': delta,
            'p_value': p_val,
            'effect_size': effect_size,
            'early_n': len(early_smi),
            'late_n': len(late_smi)
        }
    
    return results


def analyze_gap_closure(sessions_data, animal_id):
    """
    NEW: Analyze whether superficial layers "catch up" to deep layers
    """
    
    print(f"\n{'='*70}")
    print("ANALYSIS: GAP CLOSURE (Do superficial layers catch up?)")
    print(f"{'='*70}")
    
    days = sorted(sessions_data.keys())
    
    gaps = []  # Deep - Superficial difference per day
    gap_days = []
    
    for day in days:
        sup_medians = []
        deep_medians = []
        
        for layer in ['L2/3', 'L4']:
            if layer in sessions_data[day]['layers'] and sessions_data[day]['layers'][layer]['n_valid'] > 0:
                sup_medians.append(sessions_data[day]['layers'][layer]['median_smi'])
        
        for layer in ['L5', 'L6']:
            if layer in sessions_data[day]['layers'] and sessions_data[day]['layers'][layer]['n_valid'] > 0:
                deep_medians.append(sessions_data[day]['layers'][layer]['median_smi'])
        
        if sup_medians and deep_medians:
            gap = np.mean(deep_medians) - np.mean(sup_medians)
            gaps.append(gap)
            gap_days.append(day)
            print(f"  Day {day}: Deep-Superficial gap = {gap:+.3f}")
    
    if len(gaps) < 3:
        print("  Insufficient data for gap analysis")
        return None
    
    gaps = np.array(gaps)
    gap_days = np.array(gap_days)
    
    # Correlation: does gap decrease over time?
    rho, p_val = stats.spearmanr(gap_days, gaps)
    
    print(f"\n  Gap trend: ρ={rho:.3f}, p={p_val:.4f}")
    
    if rho < 0 and p_val < 0.1:
        print("  → Gap appears to DECREASE over time (superficial catching up)")
    elif rho > 0:
        print("  → Gap appears to INCREASE over time")
    else:
        print("  → No clear trend in gap")
    
    return {
        'days': gap_days.tolist(),
        'gaps': gaps.tolist(),
        'spearman_rho': rho,
        'spearman_p': p_val,
        'initial_gap': gaps[0],
        'final_gap': gaps[-1]
    }


def analyze_layer_differences_per_day(sessions_data, animal_id):
    """Layer differences within each day with pairwise comparisons."""
    
    print(f"\n{'='*70}")
    print("ANALYSIS: LAYER DIFFERENCES WITHIN EACH DAY")
    print(f"{'='*70}")
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    results = {}
    
    for day in sorted(sessions_data.keys()):
        session = sessions_data[day]
        print(f"\n--- Day {day} ---")
        
        day_results = {'medians': {}, 'n_cells': {}, 'smi_values': {}}
        
        for layer in layer_order:
            if layer in session['layers'] and session['layers'][layer]['n_valid'] > 0:
                ld = session['layers'][layer]
                day_results['medians'][layer] = ld['median_smi']
                day_results['n_cells'][layer] = ld['n_valid']
                day_results['smi_values'][layer] = ld['SMI']
                print(f"  {layer}: n={ld['n_valid']}, median={ld['median_smi']:.3f}")
        
        valid_layers = [l for l in layer_order if l in day_results['smi_values'] and len(day_results['smi_values'][l]) > 0]
        
        if len(valid_layers) >= 2:
            groups = [day_results['smi_values'][l] for l in valid_layers]
            h_stat, kw_p = stats.kruskal(*groups)
            print(f"\n  Kruskal-Wallis: H={h_stat:.3f}, p={kw_p:.4f}")
            day_results['kruskal_wallis'] = {'H': h_stat, 'p': kw_p}
            
            # Key comparison: L2/3 vs L5+L6
            if 'L2/3' in valid_layers and ('L5' in valid_layers or 'L6' in valid_layers):
                sup_smi = day_results['smi_values']['L2/3']
                deep_smi = []
                for l in ['L5', 'L6']:
                    if l in valid_layers:
                        deep_smi.extend(day_results['smi_values'][l])
                
                u_stat, p_sup_deep = stats.mannwhitneyu(sup_smi, deep_smi, alternative='two-sided')
                day_results['sup_vs_deep'] = {'U': u_stat, 'p': p_sup_deep}
                sig = '*' if p_sup_deep < 0.05 else ''
                print(f"  L2/3 vs Deep: p={p_sup_deep:.4f} {sig}")
        
        results[day] = day_results
    
    return results


def analyze_temporal_progression_per_layer(sessions_data, animal_id):
    """Temporal trends per layer."""
    
    print(f"\n{'='*70}")
    print("ANALYSIS: TEMPORAL PROGRESSION PER LAYER")
    print(f"{'='*70}")
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    results = {}
    
    for layer in layer_order:
        print(f"\n--- {layer} ---")
        
        days, medians, n_cells = [], [], []
        
        for day in sorted(sessions_data.keys()):
            if layer in sessions_data[day]['layers']:
                ld = sessions_data[day]['layers'][layer]
                if ld['n_valid'] > 0:
                    days.append(day)
                    medians.append(ld['median_smi'])
                    n_cells.append(ld['n_valid'])
                    print(f"  Day {day}: median={ld['median_smi']:.3f} (n={ld['n_valid']})")
        
        if len(days) < 3:
            print("  Insufficient data")
            continue
        
        days = np.array(days)
        medians = np.array(medians)
        
        rho, p_spearman = stats.spearmanr(days, medians)
        slope, intercept, r_val, p_linreg, _ = stats.linregress(days, medians)
        
        print(f"\n  Spearman: ρ={rho:.3f}, p={p_spearman:.4f}")
        print(f"  Linear: slope={slope:.4f}/day, R²={r_val**2:.3f}")
        
        results[layer] = {
            'days': days.tolist(),
            'medians': medians.tolist(),
            'n_cells': n_cells,
            'spearman': {'rho': rho, 'p': p_spearman},
            'linreg': {'slope': slope, 'r2': r_val**2, 'p': p_linreg}
        }
    
    return results


def create_improved_visualizations(sessions_data, layer_diff_results, temporal_results,
                                   early_late_results, gap_results, animal_id, save_path=None):
    """Create improved visualization with all analyses."""
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    fig = plt.figure(figsize=(20, 18))
    
    # =========================================================================
    # Row 1: Heatmap, Trajectory, Day 1 boxplot
    # =========================================================================
    
    # Panel 1: Heatmap
    ax1 = fig.add_subplot(3, 3, 1)
    days = sorted(sessions_data.keys())
    heatmap_data = np.full((len(layer_order), len(days)), np.nan)
    
    for j, day in enumerate(days):
        for i, layer in enumerate(layer_order):
            if layer in sessions_data[day]['layers']:
                heatmap_data[i, j] = sessions_data[day]['layers'][layer]['median_smi']
    
    im = ax1.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.8)
    ax1.set_xticks(range(len(days)))
    ax1.set_xticklabels([f'Day {d}' for d in days], fontsize=9)
    ax1.set_yticks(range(len(layer_order)))
    ax1.set_yticklabels(layer_order)
    
    for i in range(len(layer_order)):
        for j in range(len(days)):
            if not np.isnan(heatmap_data[i, j]):
                ax1.text(j, i, f'{heatmap_data[i, j]:.2f}', ha='center', va='center', fontsize=9)
    
    ax1.set_title('Median SMI by Layer × Day', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Median SMI', shrink=0.8)
    
    # Panel 2: Trajectory
    ax2 = fig.add_subplot(3, 3, 2)
    for layer in layer_order:
        if layer in temporal_results:
            tr = temporal_results[layer]
            ax2.plot(tr['days'], tr['medians'], 'o-', color=layer_colors[layer],
                    linewidth=2.5, markersize=10, label=layer)
    
    ax2.set_xlabel('Day', fontsize=11)
    ax2.set_ylabel('Median SMI', fontsize=11)
    ax2.set_title('SMI Development Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Panel 3: Day 1 boxplot with significance
    ax3 = fig.add_subplot(3, 3, 3)
    first_day = min(sessions_data.keys())
    day1_data, day1_labels, day1_colors = [], [], []
    
    for layer in layer_order:
        if layer in sessions_data[first_day]['layers']:
            smi = sessions_data[first_day]['layers'][layer]['SMI']
            if len(smi) > 0:
                day1_data.append(smi)
                day1_labels.append(f'{layer}\n(n={len(smi)})')
                day1_colors.append(layer_colors[layer])
    
    if day1_data:
        bp = ax3.boxplot(day1_data, labels=day1_labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], day1_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Add significance marker if available
        if first_day in layer_diff_results and 'kruskal_wallis' in layer_diff_results[first_day]:
            kw_p = layer_diff_results[first_day]['kruskal_wallis']['p']
            sig = '***' if kw_p < 0.001 else '**' if kw_p < 0.01 else '*' if kw_p < 0.05 else 'ns'
            ax3.text(0.95, 0.95, f'KW p={kw_p:.4f} {sig}', transform=ax3.transAxes,
                    ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white'))
    
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel('SMI', fontsize=11)
    ax3.set_title(f'Day {first_day}: Layer Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(-0.5, 1.1)
    
    # =========================================================================
    # Row 2: Last day, Sup vs Deep, Early vs Late
    # =========================================================================
    
    # Panel 4: Last day boxplot
    ax4 = fig.add_subplot(3, 3, 4)
    last_day = max(sessions_data.keys())
    lastday_data, lastday_labels, lastday_colors = [], [], []
    
    for layer in layer_order:
        if layer in sessions_data[last_day]['layers']:
            smi = sessions_data[last_day]['layers'][layer]['SMI']
            if len(smi) > 0:
                lastday_data.append(smi)
                lastday_labels.append(f'{layer}\n(n={len(smi)})')
                lastday_colors.append(layer_colors[layer])
    
    if lastday_data:
        bp = ax4.boxplot(lastday_data, labels=lastday_labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], lastday_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_ylabel('SMI', fontsize=11)
    ax4.set_title(f'Day {last_day}: Layer Comparison', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(-0.5, 1.1)
    
    # Panel 5: Superficial vs Deep with gap
    ax5 = fig.add_subplot(3, 3, 5)
    
    sup_medians, deep_medians, plot_days = [], [], []
    for day in sorted(sessions_data.keys()):
        sup_vals, deep_vals = [], []
        for layer in ['L2/3', 'L4']:
            if layer in sessions_data[day]['layers']:
                sup_vals.append(sessions_data[day]['layers'][layer]['median_smi'])
        for layer in ['L5', 'L6']:
            if layer in sessions_data[day]['layers']:
                deep_vals.append(sessions_data[day]['layers'][layer]['median_smi'])
        if sup_vals and deep_vals:
            plot_days.append(day)
            sup_medians.append(np.mean(sup_vals))
            deep_medians.append(np.mean(deep_vals))
    
    if plot_days:
        ax5.plot(plot_days, sup_medians, 'o-', color='#1E88E5', linewidth=2.5,
                markersize=10, label='Superficial (L2/3, L4)')
        ax5.plot(plot_days, deep_medians, 's-', color='#E53935', linewidth=2.5,
                markersize=10, label='Deep (L5, L6)')
        ax5.fill_between(plot_days, sup_medians, deep_medians, alpha=0.2, color='gray')
        
        # Annotate gap
        for i, (d, s, dp) in enumerate(zip(plot_days, sup_medians, deep_medians)):
            if i == 0 or i == len(plot_days) - 1:
                gap = dp - s
                ax5.annotate(f'Δ={gap:.2f}', xy=(d, (s + dp) / 2), fontsize=9,
                           ha='left' if i == 0 else 'right')
    
    ax5.set_xlabel('Day', fontsize=11)
    ax5.set_ylabel('Mean of Median SMI', fontsize=11)
    ax5.set_title('Superficial vs Deep Layers', fontsize=12, fontweight='bold')
    ax5.legend(loc='lower right')
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Early vs Late comparison (NEW)
    ax6 = fig.add_subplot(3, 3, 6)
    
    if early_late_results:
        x = np.arange(len(layer_order))
        width = 0.35
        
        early_vals = [early_late_results.get(l, {}).get('early_median', np.nan) for l in layer_order]
        late_vals = [early_late_results.get(l, {}).get('late_median', np.nan) for l in layer_order]
        
        bars1 = ax6.bar(x - width/2, early_vals, width, label='Early (Days 1-2)', color='lightblue', edgecolor='black')
        bars2 = ax6.bar(x + width/2, late_vals, width, label='Late (Days 6-7)', color='salmon', edgecolor='black')
        
        # Add significance markers
        for i, layer in enumerate(layer_order):
            if layer in early_late_results:
                p = early_late_results[layer]['p_value']
                if p < 0.001:
                    sig = '***'
                elif p < 0.01:
                    sig = '**'
                elif p < 0.05:
                    sig = '*'
                else:
                    sig = ''
                if sig:
                    max_val = max(early_vals[i], late_vals[i])
                    ax6.text(i, max_val + 0.05, sig, ha='center', fontsize=12, fontweight='bold')
        
        ax6.set_xticks(x)
        ax6.set_xticklabels(layer_order)
        ax6.set_ylabel('Median SMI', fontsize=11)
        ax6.set_title('Early vs Late: SMI Development', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
    else:
        ax6.text(0.5, 0.5, 'Insufficient data\nfor early/late comparison',
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Early vs Late Comparison', fontsize=12, fontweight='bold')
    
    # =========================================================================
    # Row 3: Gap closure, Proportion modulated, Summary
    # =========================================================================
    
    # Panel 7: Gap closure analysis (NEW)
    ax7 = fig.add_subplot(3, 3, 7)
    
    if gap_results:
        ax7.plot(gap_results['days'], gap_results['gaps'], 'ko-', linewidth=2, markersize=10)
        ax7.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax7.fill_between(gap_results['days'], 0, gap_results['gaps'], alpha=0.3,
                        color='red' if gap_results['gaps'][0] > 0 else 'blue')
        
        # Add trend line
        z = np.polyfit(gap_results['days'], gap_results['gaps'], 1)
        p = np.poly1d(z)
        ax7.plot(gap_results['days'], p(gap_results['days']), 'r--', alpha=0.7)
        
        rho = gap_results['spearman_rho']
        p_val = gap_results['spearman_p']
        ax7.text(0.95, 0.95, f'ρ={rho:.2f}, p={p_val:.3f}', transform=ax7.transAxes,
                ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white'))
    
    ax7.set_xlabel('Day', fontsize=11)
    ax7.set_ylabel('Gap (Deep - Superficial)', fontsize=11)
    ax7.set_title('Gap Closure Analysis', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # Panel 8: Proportion of modulated cells over time (NEW)
    ax8 = fig.add_subplot(3, 3, 8)
    
    for layer in layer_order:
        days_layer = []
        prop_mod = []
        for day in sorted(sessions_data.keys()):
            if layer in sessions_data[day]['layers']:
                ld = sessions_data[day]['layers'][layer]
                if ld['n_valid'] > 0:
                    days_layer.append(day)
                    prop_mod.append(ld.get('prop_modulated', np.mean(ld['SMI'] > 0.1)))
        
        if days_layer:
            ax8.plot(days_layer, prop_mod, 'o-', color=layer_colors[layer],
                    linewidth=2, markersize=8, label=layer)
    
    ax8.set_xlabel('Day', fontsize=11)
    ax8.set_ylabel('Proportion (SMI > 0.1)', fontsize=11)
    ax8.set_title('Proportion of Spatially Modulated Cells', fontsize=12, fontweight='bold')
    ax8.legend(loc='lower right')
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(0, 1)
    
    # Panel 9: Statistical summary
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    
    summary = f"STATISTICAL SUMMARY - {animal_id}\n{'='*45}\n\n"
    summary += f"Sessions: Day {min(sessions_data.keys())} to Day {max(sessions_data.keys())}\n\n"
    
    # Day 1 result
    first_day = min(sessions_data.keys())
    if first_day in layer_diff_results and 'kruskal_wallis' in layer_diff_results[first_day]:
        kw_p = layer_diff_results[first_day]['kruskal_wallis']['p']
        sig = '*' if kw_p < 0.05 else ''
        summary += f"Day {first_day} Layer Differences:\n  Kruskal-Wallis: p={kw_p:.4f} {sig}\n\n"
    
    # Temporal trends
    summary += "Temporal Trends (Spearman ρ):\n"
    for layer in layer_order:
        if layer in temporal_results:
            tr = temporal_results[layer]
            rho = tr['spearman']['rho']
            p = tr['spearman']['p']
            sig = '*' if p < 0.05 else ''
            summary += f"  {layer}: ρ={rho:+.3f}, p={p:.4f} {sig}\n"
    
    # Early vs Late
    if early_late_results:
        summary += "\nEarly vs Late (Days 1-2 vs 6-7):\n"
        for layer in layer_order:
            if layer in early_late_results:
                el = early_late_results[layer]
                sig = '*' if el['p_value'] < 0.05 else ''
                summary += f"  {layer}: Δ={el['delta']:+.3f}, p={el['p_value']:.4f} {sig}\n"
    
    # Gap closure
    if gap_results:
        summary += f"\nGap Closure:\n"
        summary += f"  Initial gap: {gap_results['initial_gap']:.3f}\n"
        summary += f"  Final gap: {gap_results['final_gap']:.3f}\n"
        summary += f"  Trend: ρ={gap_results['spearman_rho']:.3f}, p={gap_results['spearman_p']:.4f}\n"
    
    ax9.text(0.05, 0.95, summary, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig.suptitle(f'{animal_id} - Within-Animal SMI Analysis (Improved)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig_path = os.path.join(save_path, f'{animal_id}_within_animal_smi_improved.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {os.path.basename(fig_path)}")
    
    return fig


def run_within_animal_analysis(animal_dir, save_path=None):
    """Complete workflow for within-animal SMI analysis."""
    
    # Load data
    sessions_data, animal_id = load_animal_smi_data(animal_dir)
    
    if len(sessions_data) == 0:
        print("No data loaded. Exiting.")
        return None
    
    # Create save directory
    if save_path is None:
        save_path = animal_dir
    os.makedirs(save_path, exist_ok=True)
    
    # Analysis 1: Layer differences within each day
    layer_diff_results = analyze_layer_differences_per_day(sessions_data, animal_id)
    
    # Analysis 2: Temporal progression per layer
    temporal_results = analyze_temporal_progression_per_layer(sessions_data, animal_id)
    
    # Analysis 3: Early vs Late comparison (NEW)
    early_late_results = analyze_early_vs_late(sessions_data, animal_id)
    
    # Analysis 4: Gap closure (NEW)
    gap_results = analyze_gap_closure(sessions_data, animal_id)
    
    # Create visualizations
    fig = create_improved_visualizations(
        sessions_data, layer_diff_results, temporal_results,
        early_late_results, gap_results, animal_id, save_path
    )
    
    results = {
        'animal_id': animal_id,
        'sessions_data': sessions_data,
        'layer_differences': layer_diff_results,
        'temporal_progression': temporal_results,
        'early_vs_late': early_late_results,
        'gap_closure': gap_results,
        'figure': fig
    }
    
    print(f"\n{'='*70}")
    print("WITHIN-ANIMAL ANALYSIS COMPLETE")
    print(f"{'='*70}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    animal_dir = r"D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging"
    save_dir = r"D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\smi_analysis2"
    
    results = run_within_animal_analysis(animal_dir, save_dir)
    
    plt.show()