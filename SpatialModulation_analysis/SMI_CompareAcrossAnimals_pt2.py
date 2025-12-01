"""
SMI_CompareAcrossAnimals_pt2.py (IMPROVED) - PART 2
Visualizations - Run after Part 1

JSY, 2025
"""
import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle

# Layer configuration
LAYER_ORDER = ['L2/3', 'L4', 'L5', 'L6']
LAYER_COLORS = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}


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
        
        ax6.text(0.95, 0.95, f"ρ={gap['spearman_rho']:.2f}\np={gap['spearman_p']:.3f}",
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
            summary += f"  {layer}: Δ={el['delta']:+.3f}, p={el['p_value']:.3f}, d={el['cohens_d']:.2f} {sig}\n"
    
    # Gap closure
    if gap:
        summary += f"\nGAP CLOSURE:\n"
        summary += f"  Initial: {gap['initial_gap']:.3f}, Final: {gap['final_gap']:.3f}\n"
        summary += f"  Trend: ρ={gap['spearman_rho']:.2f}, p={gap['spearman_p']:.3f}\n"
    
    ax9.text(0.05, 0.95, summary, transform=ax9.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig.suptitle('Across-Animals SMI Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig_path = os.path.join(save_path, 'across_animals_smi_analysis.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {fig_path}")
    
    return fig


def run_visualizations(results):
    """Run all visualizations using results from Part 1."""
    
    print(f"\n{'='*80}")
    print("ACROSS-ANIMALS SMI ANALYSIS - PART 2 (VISUALIZATIONS)")
    print(f"{'='*80}")
    
    save_path = results.get('save_path', None)
    
    fig = create_main_figure(results, save_path)
    
    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*70}")
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    save_dir = r"D:\V1_SpatialModulation\2p\V1_prism\across_animals_smi_analysis"
    
    # Load results from Part 1
    pkl_path = os.path.join(save_dir, 'analysis_results.pkl')
    
    if os.path.exists(pkl_path):
        print(f"Loading results from Part 1: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        
        fig = run_visualizations(results)
        plt.show()
    else:
        print(f"ERROR: Run Part 1 first to generate {pkl_path}")