"""
SpeedModulation_analyze_across_sessions.py

Compare speed modulation across all recording sessions for JSY052.
"""
import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
import pandas as pd
from helper import files

# =============================================================================
# CONFIGURATION
# =============================================================================
animal_id = "JSY052"
base_path = r"F:\2P\spmod\JSY052_ChrnoicImaging"

# List all session paths (UPDATE with your actual paths)
sessions = [
    {
        'day': 1,
        'date': '251009',
        'path': os.path.join(base_path, '251009_JSY_JSY052_SpatialModulation_Day1\TSeries-10092025-1542-002/speed_tuning_analysis')
    },
    {
        'day': 2,
        'date': '251010',
        'path': os.path.join(base_path, '251010_JSY_JSY052_SpatialModulation_Day2\TSeries-10102025-0916-001/speed_tuning_analysis')
    },
    {
        'day': 3,
        'date': '251011',
        'path': os.path.join(base_path, '251011_JSY_JSY052_SpatialModulation_Day3\TSeries-10112025-1441-002/speed_tuning_analysis')
    },
    {
        'day': 4,
        'date': '251012',
        'path': os.path.join(base_path, '251012_JSY_JSY052_SpatialModulation_Day4\TSeries-10122025-1212-002/speed_tuning_analysis')
    },
    {
        'day': 5,
        'date': '251013',
        'path': os.path.join(base_path, '251013_JSY_JSY052_SpatialModulation_Day5\TSeries-10132025-1236-001/speed_tuning_analysis')
    }
]

output_dir = os.path.join(base_path, f'{animal_id}_CrossSession_Analysis2')
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# LOAD ALL SESSIONS
# =============================================================================
print("="*80)
print("LOADING ALL SESSIONS")
print("="*80)

session_data = []

for session in sessions:
    detailed_file = os.path.join(session['path'], 'speed_modulation_DETAILED.h5')
    
    if not os.path.exists(detailed_file):
        print(f"⚠️  Missing: Day {session['day']} - {detailed_file}")
        continue
    
    data = files.read_h5(detailed_file)
    data['day'] = session['day']
    data['date'] = session['date']
    session_data.append(data)
    
    print(f"✓ Loaded Day {session['day']}: {data['n_reliable_cells']} reliable cells")

print(f"\n✓ Total sessions loaded: {len(session_data)}")

# =============================================================================
# ANALYSIS 1: OVERALL TRENDS ACROSS DAYS
# =============================================================================
print("\n" + "="*80)
print("ANALYSIS 1: OVERALL TRENDS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Panel A: Overall % modulated
days = [s['day'] for s in session_data]
overall_pct = [s['overall_prop_modulated'] * 100 for s in session_data]

axes[0, 0].plot(days, overall_pct, 'o-', linewidth=3, markersize=10, color='#2E7D32')
axes[0, 0].set_xlabel('Day', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('% Speed-Modulated', fontsize=12, fontweight='bold')
axes[0, 0].set_title('A. Overall Modulation Across Days', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(days)

# Panel B: Mean MI across days
mean_MIs = [s['overall_mean_mod_index'] for s in session_data]

axes[0, 1].plot(days, mean_MIs, 'o-', linewidth=3, markersize=10, color='#D32F2F')
axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Day', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Mean Modulation Index', fontsize=12, fontweight='bold')
axes[0, 1].set_title('B. Mean MI Across Days', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(days)

# Panel C: Positive vs Negative
n_pos = [s['overall_n_positive'] for s in session_data]
n_neg = [s['overall_n_negative'] for s in session_data]

x = np.arange(len(days))
width = 0.35

axes[0, 2].bar(x - width/2, n_pos, width, label='Positive', color='red', alpha=0.7)
axes[0, 2].bar(x + width/2, n_neg, width, label='Negative', color='blue', alpha=0.7)
axes[0, 2].set_xlabel('Day', fontsize=12, fontweight='bold')
axes[0, 2].set_ylabel('Number of Cells', fontsize=12, fontweight='bold')
axes[0, 2].set_title('C. Positive vs Negative', fontsize=13, fontweight='bold')
axes[0, 2].set_xticks(x)
axes[0, 2].set_xticklabels(days)
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Panel D: Speed sampling (frames per category)
n_slow = [s['n_frames_slow'] for s in session_data]
n_medium = [s['n_frames_medium'] for s in session_data]
n_fast = [s['n_frames_fast'] for s in session_data]

axes[1, 0].plot(days, n_slow, 'o-', label='Slow (2-10)', linewidth=2, markersize=8)
axes[1, 0].plot(days, n_medium, 's-', label='Medium (10-20)', linewidth=2, markersize=8)
axes[1, 0].plot(days, n_fast, '^-', label='Fast (>20)', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Day', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Number of Frames', fontsize=12, fontweight='bold')
axes[1, 0].set_title('D. Speed Sampling Across Days', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks(days)

# Panel E: Chi-square statistics
chi2_vals = [s.get('chi2_statistic', np.nan) for s in session_data]
p_vals = [s.get('chi2_p_value', np.nan) for s in session_data]

axes[1, 1].bar(days, chi2_vals, alpha=0.7, color='purple')
axes[1, 1].set_xlabel('Day', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('χ² Statistic', fontsize=12, fontweight='bold')
axes[1, 1].set_title('E. Layer Difference Significance', fontsize=13, fontweight='bold')
axes[1, 1].set_xticks(days)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add p-values as text
for i, (d, p) in enumerate(zip(days, p_vals)):
    if not np.isnan(p):
        sig_text = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        axes[1, 1].text(d, chi2_vals[i], sig_text, ha='center', va='bottom', fontweight='bold')

# Panel F: Sample sizes
n_cells = [s['n_reliable_cells'] for s in session_data]
n_laps = [s['n_laps'] for s in session_data]

ax_f = axes[1, 2]
ax_f.plot(days, n_cells, 'o-', label='Reliable Cells', linewidth=2, markersize=8, color='green')
ax_f.set_xlabel('Day', fontsize=12, fontweight='bold')
ax_f.set_ylabel('Number of Cells', fontsize=12, fontweight='bold', color='green')
ax_f.tick_params(axis='y', labelcolor='green')
ax_f.set_title('F. Sample Sizes', fontsize=13, fontweight='bold')
ax_f.set_xticks(days)
ax_f.grid(True, alpha=0.3)

ax_f2 = ax_f.twinx()
ax_f2.plot(days, n_laps, 's-', label='Laps', linewidth=2, markersize=8, color='orange')
ax_f2.set_ylabel('Number of Laps', fontsize=12, fontweight='bold', color='orange')
ax_f2.tick_params(axis='y', labelcolor='orange')

plt.suptitle(f'{animal_id} - Overall Speed Modulation Across Sessions', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'CrossSession_Overall.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'CrossSession_Overall.pdf'), bbox_inches='tight')

print("✓ Overall trends figure saved")

# =============================================================================
# ANALYSIS 2: LAYER-SPECIFIC TRENDS
# =============================================================================
print("\n" + "="*80)
print("ANALYSIS 2: LAYER-SPECIFIC TRENDS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors_layer = {'L2/3': '#4472C4', 'L4': '#ED7D31', 'L5': '#70AD47', 'L6': '#C5504B'}

# Extract layer data
layer_data = {'L2/3': {}, 'L4': {}, 'L5': {}, 'L6': {}}

for layer in ['L2/3', 'L4', 'L5', 'L6']:
    prefix = layer.replace('/', '_')
    layer_data[layer]['days'] = []
    layer_data[layer]['prop'] = []
    layer_data[layer]['mean_mi'] = []
    layer_data[layer]['n_pos'] = []
    layer_data[layer]['n_neg'] = []
    
    for s in session_data:
        if f'{prefix}_n_total' in s:
            layer_data[layer]['days'].append(s['day'])
            layer_data[layer]['prop'].append(s[f'{prefix}_prop_speed_mod'] * 100)
            layer_data[layer]['mean_mi'].append(s[f'{prefix}_mean_mod_index'])
            layer_data[layer]['n_pos'].append(s[f'{prefix}_n_positive'])
            layer_data[layer]['n_neg'].append(s[f'{prefix}_n_negative'])

# Panel A: % Modulated by layer
for layer, data in layer_data.items():
    if len(data['days']) > 0:
        axes[0, 0].plot(data['days'], data['prop'], 'o-', 
                       label=layer, color=colors_layer[layer],
                       linewidth=2, markersize=8)

axes[0, 0].set_xlabel('Day', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('% Speed-Modulated', fontsize=12, fontweight='bold')
axes[0, 0].set_title('A. Layer-Specific Modulation', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(days)

# Panel B: Mean MI by layer
for layer, data in layer_data.items():
    if len(data['days']) > 0:
        axes[0, 1].plot(data['days'], data['mean_mi'], 'o-',
                       label=layer, color=colors_layer[layer],
                       linewidth=2, markersize=8)

axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Day', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Mean Modulation Index', fontsize=12, fontweight='bold')
axes[0, 1].set_title('B. Mean MI by Layer', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(days)

# Panel C: L6 detailed trajectory
if len(layer_data['L6']['days']) > 0:
    ax_c = axes[1, 0]
    ax_c.plot(layer_data['L6']['days'], layer_data['L6']['mean_mi'], 
             'o-', color=colors_layer['L6'], linewidth=3, markersize=12)
    ax_c.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax_c.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax_c.set_ylabel('L6 Mean MI', fontsize=12, fontweight='bold')
    ax_c.set_title('C. Layer 6 MI Development', fontsize=13, fontweight='bold')
    ax_c.grid(True, alpha=0.3)
    ax_c.set_xticks(layer_data['L6']['days'])
    
    # Add values on points
    for d, mi in zip(layer_data['L6']['days'], layer_data['L6']['mean_mi']):
        ax_c.text(d, mi, f'{mi:.3f}', ha='center', va='bottom', fontweight='bold')

# Panel D: Positive/Negative ratio by layer
ax_d = axes[1, 1]
x_pos = np.arange(len(days))
width = 0.2

for i, layer in enumerate(['L2/3', 'L4', 'L5', 'L6']):
    data = layer_data[layer]
    if len(data['days']) > 0:
        # Calculate % negative
        pct_neg = []
        for pos, neg in zip(data['n_pos'], data['n_neg']):
            total = pos + neg
            pct_neg.append((neg / total * 100) if total > 0 else 0)
        
        ax_d.bar(x_pos + i*width, pct_neg, width, 
                label=layer, color=colors_layer[layer], alpha=0.7)

ax_d.axhline(50, color='black', linestyle='--', alpha=0.5, label='50% (balanced)')
ax_d.set_xlabel('Day', fontsize=12, fontweight='bold')
ax_d.set_ylabel('% Negative Modulation', fontsize=12, fontweight='bold')
ax_d.set_title('D. Negative Modulation by Layer', fontsize=13, fontweight='bold')
ax_d.set_xticks(x_pos + width*1.5)
ax_d.set_xticklabels(days)
ax_d.legend()
ax_d.grid(True, alpha=0.3, axis='y')
ax_d.set_ylim([0, 100])

plt.suptitle(f'{animal_id} - Layer-Specific Speed Modulation Development', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'CrossSession_Layers.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'CrossSession_Layers.pdf'), bbox_inches='tight')

print("✓ Layer-specific trends figure saved")

# =============================================================================
# SAVE SUMMARY TABLE
# =============================================================================
print("\n" + "="*80)
print("CREATING SUMMARY TABLE")
print("="*80)

# Create pandas DataFrame
summary_data = []

for s in session_data:
    row = {
        'Day': s['day'],
        'Date': s['date'],
        'N_Cells': s['n_reliable_cells'],
        'N_Laps': s['n_laps'],
        'N_Frames': s['n_frames'],
        'Frames_Slow': s['n_frames_slow'],
        'Frames_Fast': s['n_frames_fast'],
        'Overall_Pct': f"{s['overall_prop_modulated']*100:.1f}%",
        'Overall_MI': f"{s['overall_mean_mod_index']:.3f}",
        'N_Positive': s['overall_n_positive'],
        'N_Negative': s['overall_n_negative'],
    }
    
    # Add layer data
    for layer in ['L2/3', 'L4', 'L5', 'L6']:
        prefix = layer.replace('/', '_')
        if f'{prefix}_n_total' in s:
            row[f'{layer}_Pct'] = f"{s[f'{prefix}_prop_speed_mod']*100:.1f}%"
            row[f'{layer}_MI'] = f"{s[f'{prefix}_mean_mod_index']:.3f}"
    
    summary_data.append(row)

df = pd.DataFrame(summary_data)

# Save to CSV
csv_file = os.path.join(output_dir, f'{animal_id}_CrossSession_Summary.csv')
df.to_csv(csv_file, index=False)
print(f"✓ Summary table saved to: {csv_file}")

# Print table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(df.to_string(index=False))

print("\n" + "="*80)
print("CROSS-SESSION ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll outputs saved to: {output_dir}")