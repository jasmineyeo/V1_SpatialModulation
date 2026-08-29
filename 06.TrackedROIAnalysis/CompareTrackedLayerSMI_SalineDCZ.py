"""
CompareTrackedLayerSMI_SalineDCZ.py
Paired, same-cell, layer-specific SMI comparison between two drug
conditions (e.g. SALINE vs DCZ) for cells tracked across sessions by
TrackROIs_SalineDCZ.py.

Pipeline this sits on top of:
  1. SMI_FullSession_Interactive.py -> smi_results.h5 per session
  2. TrackROIs_SalineDCZ.py         -> roi_tracking_results.h5
  3. MergeTrackedLayerSMI.py        -> tidy tracked_cell_row x day_label table
  4. THIS SCRIPT                    -> per-layer paired stats + figure

Analysis choices (per JSY, 2026):
  - Uses SMI_layer_reliable (the reliability+validity-filtered SMI used
    everywhere else in this pipeline), not SMI_global.
  - Tracked cells whose layer assignment disagrees between the two
    sessions (boundary drift) are excluded from the layer-specific
    comparison rather than arbitrarily assigned to one session's call.
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['legend.fontsize'] = 14
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
from scipy import stats

from MergeTrackedLayerSMI import merge_tracked_layer_smi

LAYER_ORDER = ['L2/3', 'L4', 'L5', 'L6']
LAYER_COLORS = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}


# ============================================================
# CONFIGURATION
# ============================================================
tracking_h5_path = (
    r"D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260724_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_1\TrackedROIs\roi_tracking_results.h5")
required_days = ['SALINE', 'DCZ']
smi_col = 'SMI_layer_reliable'  # or 'SMI_global'


# ============================================================
# Function 1: Load the merged table and pivot to one row per tracked cell
# ============================================================
def load_and_pivot(tracking_h5_path, required_days, smi_col='SMI_layer_reliable'):
    day_a, day_b = required_days

    df = merge_tracked_layer_smi(tracking_h5_path, required_days=list(required_days))

    wide = df.pivot(index='tracked_cell_row', columns='day_label')
    wide.columns = [f'{col}_{day}' for col, day in wide.columns]
    wide = wide.reset_index()

    layer_a_col, layer_b_col = f'layer_{day_a}', f'layer_{day_b}'
    smi_a_col, smi_b_col = f'{smi_col}_{day_a}', f'{smi_col}_{day_b}'

    n_total = len(wide)

    mismatch = wide[layer_a_col] != wide[layer_b_col]
    n_mismatch = int(mismatch.sum())
    print(f"{n_mismatch}/{n_total} tracked cells changed layer assignment between "
          f"{day_a} and {day_b} — excluding from layer-specific comparison.")
    wide = wide[~mismatch].copy()
    wide['layer'] = wide[layer_a_col]

    valid = wide[smi_a_col].notna() & wide[smi_b_col].notna()
    n_dropped = int((~valid).sum())
    print(f"{n_dropped}/{len(wide)} remaining cells dropped — not reliable+valid "
          f"(non-NaN {smi_col}) in both {day_a} and {day_b}.")
    wide = wide[valid].copy()

    wide['delta_smi'] = wide[smi_b_col] - wide[smi_a_col]

    print(f"\nFinal comparable cells: {len(wide)}/{n_total}")
    return wide, day_a, day_b, smi_a_col, smi_b_col


# ============================================================
# Function 2: Per-layer paired stats
# ============================================================
def compare_layers(wide, day_a, day_b, smi_a_col, smi_b_col):
    print("\n" + "=" * 60)
    print(f"PAIRED LAYER-SPECIFIC SMI COMPARISON: {day_a} vs {day_b}")
    print("=" * 60)

    results = {}
    layers_present = [l for l in LAYER_ORDER if l in wide['layer'].unique()]

    for layer in layers_present:
        sub = wide[wide['layer'] == layer]
        a = sub[smi_a_col].values
        b = sub[smi_b_col].values

        if len(sub) < 2:
            print(f"\n{layer}: n={len(sub)} — skipping (need >=2 paired cells for Wilcoxon)")
            continue

        try:
            stat, p = stats.wilcoxon(a, b)
        except ValueError as e:
            stat, p = np.nan, np.nan
            print(f"\n{layer}: Wilcoxon failed ({e})")

        median_a, median_b = np.median(a), np.median(b)
        median_delta = np.median(b - a)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        print(f"\n{layer}: n={len(sub)}")
        print(f"  median {day_a} = {median_a:.3f}, median {day_b} = {median_b:.3f}")
        print(f"  median Δ ({day_b}-{day_a}) = {median_delta:+.3f}")
        print(f"  Wilcoxon signed-rank: W={stat:.3f}, p={p:.4f} {sig}")

        results[layer] = {
            'n': len(sub),
            f'median_{day_a}': median_a,
            f'median_{day_b}': median_b,
            'median_delta': median_delta,
            'wilcoxon_stat': stat,
            'p_value': p,
        }

    return results


# ============================================================
# Function 3: Visualize — paired per-cell lines + delta summary
# ============================================================
def plot_comparison(wide, results, day_a, day_b, smi_a_col, smi_b_col, save_path=None):
    layers_present = [l for l in LAYER_ORDER if l in results]
    n_layers = len(layers_present)

    fig, axes = plt.subplots(1, n_layers + 1, figsize=(4.5 * (n_layers + 1), 5))
    if n_layers == 0:
        axes = [axes]

    for i, layer in enumerate(layers_present):
        ax = axes[i]
        sub = wide[wide['layer'] == layer]
        color = LAYER_COLORS.get(layer, 'gray')

        for _, row in sub.iterrows():
            ax.plot([0, 1], [row[smi_a_col], row[smi_b_col]],
                     '-', color=color, alpha=0.3, linewidth=1)
        ax.scatter(np.zeros(len(sub)), sub[smi_a_col], color=color, alpha=0.7, s=25, zorder=3)
        ax.scatter(np.ones(len(sub)), sub[smi_b_col], color=color, alpha=0.7, s=25, zorder=3)

        med_a = results[layer][f'median_{day_a}']
        med_b = results[layer][f'median_{day_b}']
        ax.plot([0, 1], [med_a, med_b], 'k-', linewidth=2.5, zorder=4)
        ax.scatter([0, 1], [med_a, med_b], color='black', s=80, zorder=5, marker='D')

        p = results[layer]['p_value']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.set_xticks([0, 1])
        ax.set_xticklabels([day_a, day_b])
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylabel('SMI')
        ax.set_title(f"{layer} (n={results[layer]['n']})\np={p:.4f} {sig}", fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')

    # Summary panel: median delta per layer with significance
    ax_summary = axes[-1]
    deltas = [results[l]['median_delta'] for l in layers_present]
    colors = [LAYER_COLORS.get(l, 'gray') for l in layers_present]
    bars = ax_summary.bar(layers_present, deltas, color=colors, alpha=0.8, edgecolor='black')
    ax_summary.axhline(0, color='gray', linestyle='--', alpha=0.7)
    for l, bar in zip(layers_present, bars):
        p = results[l]['p_value']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        height = bar.get_height()
        ax_summary.text(bar.get_x() + bar.get_width() / 2, height,
                        sig, ha='center', va='bottom' if height >= 0 else 'top', fontsize=12)
    ax_summary.set_ylabel(f'Median Δ SMI ({day_b} - {day_a})')
    ax_summary.set_title('Summary: SMI change by layer', fontsize=13)
    ax_summary.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Tracked-cell layer-specific SMI: {day_a} vs {day_b}',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")

    return fig


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    save_dir = os.path.dirname(tracking_h5_path)

    wide, day_a, day_b, smi_a_col, smi_b_col = load_and_pivot(
        tracking_h5_path, required_days, smi_col=smi_col
    )

    wide.to_csv(os.path.join(save_dir, 'tracked_layer_smi_comparison.csv'), index=False)
    print(f"\nPer-cell comparison table saved to "
          f"{os.path.join(save_dir, 'tracked_layer_smi_comparison.csv')}")

    results = compare_layers(wide, day_a, day_b, smi_a_col, smi_b_col)

    fig = plot_comparison(
        wide, results, day_a, day_b, smi_a_col, smi_b_col,
        save_path=os.path.join(save_dir, 'tracked_layer_smi_comparison.png')
    )
    plt.show()
