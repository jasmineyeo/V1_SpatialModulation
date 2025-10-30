"""
SpeedTuningAnalysis.py

Comprehensive speed modulation analysis for V1 neurons during virtual navigation.
Implements:
1. Simple permutation-based method (RECOMMENDED - finds 37% modulated cells)
2. Saleem et al. (2013) Q_S method (for comparison - finds 0.4% modulated cells)

The simple method is recommended for datasets with:
- Fewer trials (<100 laps)
- Negative speed modulation
- Trial-to-trial variability

JSY, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import os


class SpeedTuningAnalysis:
    """
    Comprehensive analysis of speed modulation in neural data.
    
    Methods available:
    ------------------
    1. simple_speed_modulation() - Permutation-based method (RECOMMENDED)
    2. analyze_simple_by_layer() - Simple method with layer analysis
    3. plot_simple_results() - Publication-quality figures for simple method
    4. analyze_speed_tuning_by_layer() - Saleem Q_S method (for comparison)
    5. compare_methods() - Compare both methods side-by-side
    """
    
    # =========================================================================
    # METHOD 1: SIMPLE PERMUTATION-BASED (RECOMMENDED)
    # =========================================================================
    
    @staticmethod
    def simple_speed_modulation(spike_data, speed_data, reliable_cells,
                                slow_range=(2, 10), medium_range=(10, 20), 
                                fast_range=(20, 50), min_frames_per_bin=50,
                                mod_index_threshold=0.1, p_threshold=0.05,
                                n_permutations=1000):
        """
        Simple, robust speed modulation analysis using permutation testing.
        
        RECOMMENDED METHOD: Works better than Q_S for smaller datasets.
        
        Parameters:
        -----------
        spike_data : numpy.ndarray
            Neural activity (n_cells × n_frames)
        speed_data : numpy.ndarray
            Speed at each frame (n_frames,)
        reliable_cells : numpy.ndarray
            Boolean mask of reliable cells
        slow_range : tuple
            Speed range for "slow" category (cm/s)
        medium_range : tuple
            Speed range for "medium" category (cm/s)
        fast_range : tuple
            Speed range for "fast" category (cm/s)
        min_frames_per_bin : int
            Minimum frames required per speed bin
        mod_index_threshold : float
            Minimum |MI| to be considered modulated
        p_threshold : float
            P-value threshold for significance
        n_permutations : int
            Number of permutations for statistical testing
            
        Returns:
        --------
        results : dict
            Complete results including modulation indices, p-values, etc.
        """
        print("="*80)
        print("SIMPLE SPEED MODULATION ANALYSIS")
        print("="*80)
        print(f"Method: Permutation-based modulation index")
        print(f"Permutations: {n_permutations}")
        print(f"Significance: p < {p_threshold}, |MI| > {mod_index_threshold}")
        
        # Create speed masks
        slow_mask = (speed_data >= slow_range[0]) & (speed_data < slow_range[1])
        medium_mask = (speed_data >= medium_range[0]) & (speed_data < medium_range[1])
        fast_mask = (speed_data >= fast_range[0]) & (speed_data < fast_range[1])
        
        n_slow = np.sum(slow_mask)
        n_medium = np.sum(medium_mask)
        n_fast = np.sum(fast_mask)
        
        print(f"\nSpeed categories:")
        print(f"  Slow ({slow_range[0]}-{slow_range[1]} cm/s): {n_slow} frames")
        print(f"  Medium ({medium_range[0]}-{medium_range[1]} cm/s): {n_medium} frames")
        print(f"  Fast ({fast_range[0]}-{fast_range[1]} cm/s): {n_fast} frames")
        
        if n_slow < min_frames_per_bin or n_fast < min_frames_per_bin:
            print(f"\n⚠️  WARNING: Insufficient data!")
            print(f"  Need ≥{min_frames_per_bin} frames per bin")
            return None
        
        # Analyze each cell
        reliable_indices = np.where(reliable_cells)[0]
        n_reliable = len(reliable_indices)
        
        modulation_indices = []
        p_values = []
        speed_modulated_cells = []
        mean_activities_all = []
        modulation_directions = []
        
        print(f"\nAnalyzing {n_reliable} reliable cells...")
        
        for cell_idx in tqdm(reliable_indices):
            # Mean activity in each speed bin
            mean_slow = np.mean(spike_data[cell_idx, slow_mask])
            mean_medium = np.mean(spike_data[cell_idx, medium_mask]) if n_medium >= min_frames_per_bin else mean_slow
            mean_fast = np.mean(spike_data[cell_idx, fast_mask])
            
            mean_activities_all.append([mean_slow, mean_medium, mean_fast])
            
            # Modulation index: (fast - slow) / (fast + slow)
            if (mean_fast + mean_slow) > 0:
                mod_index = (mean_fast - mean_slow) / (mean_fast + mean_slow)
            else:
                mod_index = 0
            
            modulation_indices.append(mod_index)
            
            # Permutation test
            all_activity = spike_data[cell_idx, slow_mask | fast_mask]
            perm_diffs = []
            
            for _ in range(n_permutations):
                shuffled = np.random.permutation(all_activity)
                perm_slow = np.mean(shuffled[:n_slow])
                perm_fast = np.mean(shuffled[n_slow:n_slow+n_fast])
                perm_diffs.append(perm_fast - perm_slow)
            
            real_diff = mean_fast - mean_slow
            p_value = np.sum(np.abs(perm_diffs) >= np.abs(real_diff)) / n_permutations
            p_values.append(p_value)
            
            # Determine if significantly modulated
            if p_value < p_threshold and np.abs(mod_index) > mod_index_threshold:
                speed_modulated_cells.append(cell_idx)
                modulation_directions.append('positive' if mod_index > 0 else 'negative')
            else:
                modulation_directions.append('none')
        
        modulation_indices = np.array(modulation_indices)
        p_values = np.array(p_values)
        mean_activities_all = np.array(mean_activities_all)
        
        # Count modulation types
        n_positive = np.sum(np.array([modulation_indices[list(reliable_indices).index(c)] 
                                     for c in speed_modulated_cells]) > 0)
        n_negative = len(speed_modulated_cells) - n_positive
        
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"Speed-modulated: {len(speed_modulated_cells)}/{n_reliable} ({len(speed_modulated_cells)/n_reliable*100:.1f}%)")
        print(f"  Positive modulation: {n_positive}")
        print(f"  Negative modulation: {n_negative}")
        print(f"Mean MI: {np.mean(modulation_indices):.3f}")
        print(f"Median MI: {np.median(modulation_indices):.3f}")
        
        return {
            'method': 'simple_permutation',
            'speed_modulated_cells': speed_modulated_cells,
            'reliable_indices': reliable_indices,
            'modulation_indices': modulation_indices,
            'p_values': p_values,
            'modulation_directions': modulation_directions,
            'mean_activities': mean_activities_all,
            'n_total': n_reliable,
            'n_modulated': len(speed_modulated_cells),
            'n_positive': n_positive,
            'n_negative': n_negative,
            'prop_modulated': len(speed_modulated_cells) / n_reliable,
            'mean_mod_index': np.mean(modulation_indices),
            'median_mod_index': np.median(modulation_indices),
            'speed_bins': {
                'slow': slow_range,
                'medium': medium_range,
                'fast': fast_range
            },
            'parameters': {
                'n_permutations': n_permutations,
                'p_threshold': p_threshold,
                'mod_index_threshold': mod_index_threshold,
                'min_frames_per_bin': min_frames_per_bin
            }
        }
    
    @staticmethod
    def analyze_simple_by_layer(spike_data, speed_data, layer_cells, reliable_cells,
                                slow_range=(2, 10), medium_range=(10, 20), 
                                fast_range=(20, 50), **kwargs):
        """
        Run simple speed modulation analysis separately for each layer.
        
        Parameters:
        -----------
        spike_data : numpy.ndarray
            Neural activity (n_cells × n_frames)
        speed_data : numpy.ndarray
            Speed data (n_frames,)
        layer_cells : dict
            Cell indices for each layer {'L2/3': [...], 'L4': [...], ...}
        reliable_cells : numpy.ndarray
            Boolean mask of reliable cells
        slow_range, medium_range, fast_range : tuple
            Speed ranges for binning
        **kwargs : additional arguments for simple_speed_modulation
            
        Returns:
        --------
        layer_results : dict
            Results for each layer including overall and layer-specific statistics
        """
        print("\n" + "="*80)
        print("LAYER-SPECIFIC SIMPLE SPEED MODULATION ANALYSIS")
        print("="*80)
        
        # First run overall analysis
        overall_results = SpeedTuningAnalysis.simple_speed_modulation(
            spike_data, speed_data, reliable_cells,
            slow_range, medium_range, fast_range, **kwargs
        )
        
        if overall_results is None:
            return None
        
        # Analyze by layer
        layer_results = {}
        reliable_indices = overall_results['reliable_indices']
        modulation_indices = overall_results['modulation_indices']
        speed_modulated_cells = overall_results['speed_modulated_cells']
        
        for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
            if layer_name not in layer_cells:
                continue
            
            layer_indices = layer_cells[layer_name]
            
            # Find reliable cells in this layer
            reliable_layer_cells = np.intersect1d(reliable_indices, layer_indices)
            
            if len(reliable_layer_cells) == 0:
                continue
            
            # Get speed-modulated cells in this layer
            layer_speed_mod = [c for c in speed_modulated_cells if c in reliable_layer_cells]
            
            # Calculate statistics
            layer_mod_indices = []
            n_positive = 0
            n_negative = 0
            
            for cell_idx in layer_speed_mod:
                rel_idx = np.where(reliable_indices == cell_idx)[0][0]
                mod_idx = modulation_indices[rel_idx]
                layer_mod_indices.append(mod_idx)
                
                if mod_idx > 0:
                    n_positive += 1
                else:
                    n_negative += 1
            
            mean_mod_idx = np.mean(layer_mod_indices) if len(layer_mod_indices) > 0 else 0
            
            layer_results[layer_name] = {
                'n_total': len(reliable_layer_cells),
                'n_speed_mod': len(layer_speed_mod),
                'prop_speed_mod': len(layer_speed_mod) / len(reliable_layer_cells),
                'n_positive': n_positive,
                'n_negative': n_negative,
                'mean_mod_index': mean_mod_idx,
                'speed_modulated_cells': layer_speed_mod,
                'mod_indices': layer_mod_indices
            }
            
            print(f"\n{layer_name}:")
            print(f"  Total reliable: {len(reliable_layer_cells)}")
            print(f"  Speed-modulated: {len(layer_speed_mod)} ({len(layer_speed_mod)/len(reliable_layer_cells)*100:.1f}%)")
            print(f"  Positive: {n_positive}, Negative: {n_negative}")
            print(f"  Mean MI: {mean_mod_idx:.3f}")
        
        # Statistical comparison
        print("\n" + "="*80)
        print("STATISTICAL COMPARISON ACROSS LAYERS")
        print("="*80)
        
        observed = []
        for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
            if layer_name in layer_results:
                lr = layer_results[layer_name]
                observed.append([lr['n_speed_mod'], lr['n_total'] - lr['n_speed_mod']])
        
        if len(observed) >= 2:
            observed = np.array(observed)
            chi2, p_value = stats.chi2_contingency(observed)[:2]
            print(f"\nChi-square test (proportion modulated):")
            print(f"  χ² = {chi2:.3f}, p = {p_value:.4f}")
            if p_value < 0.05:
                print("  *** Significant difference across layers")
        
        return {
            'overall': overall_results,
            'layer_results': layer_results,
            'chi2': chi2 if len(observed) >= 2 else None,
            'p_value': p_value if len(observed) >= 2 else None
        }
    
    @staticmethod
    def plot_simple_results(results, spike_data=None, speed_data=None, save_dir=None):
        """
        Create comprehensive publication-quality figures for simple method results.
        
        Parameters:
        -----------
        results : dict
            Output from analyze_simple_by_layer()
        spike_data : numpy.ndarray, optional
            Neural activity for plotting example cells
        speed_data : numpy.ndarray, optional
            Speed data for plotting example cells
        save_dir : str, optional
            Directory to save figures
        """
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        
        overall_results = results['overall']
        layer_results = results['layer_results']
        
        # Figure 1: Main results - Layer gradient
        fig1 = SpeedTuningAnalysis._plot_layer_gradient(layer_results, save_dir)
        
        # Figure 2: Detailed analysis
        fig2 = SpeedTuningAnalysis._plot_detailed_analysis(
            overall_results, layer_results, save_dir
        )
        
        # Figure 3: Example cells (if data provided)
        if spike_data is not None and speed_data is not None:
            fig3 = SpeedTuningAnalysis._plot_example_cells_simple(
                overall_results, spike_data, speed_data, save_dir
            )
        else:
            fig3 = None
        
        plt.show()
        
        return fig1, fig2, fig3
    
    @staticmethod
    def _plot_layer_gradient(layer_results, save_dir=None):
        """
        Plot main figure showing layer-specific speed modulation gradient.
        Publication-quality 3-panel figure.
        """
        fig = plt.figure(figsize=(15, 4))
        gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        layers = []
        props = []
        n_pos = []
        n_neg = []
        mean_MIs = []
        colors = ['#4472C4', '#ED7D31', '#70AD47', '#C5504B']
        
        for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
            if layer_name in layer_results:
                lr = layer_results[layer_name]
                layers.append(layer_name)
                props.append(lr['prop_speed_mod'] * 100)
                n_pos.append(lr['n_positive'])
                n_neg.append(lr['n_negative'])
                mean_MIs.append(lr['mean_mod_index'])
        
        # Panel A: Proportion modulated by layer
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(layers, props, color=colors[:len(layers)], alpha=0.8, 
                      edgecolor='black', linewidth=2)
        ax1.set_ylabel('% Speed-Modulated', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Cortical Layer', fontsize=14, fontweight='bold')
        ax1.set_title('A. Layer-Specific Speed Modulation', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, max(props) * 1.2])
        ax1.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for bar, val in zip(bars, props):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Panel B: Positive vs Negative by layer
        ax2 = fig.add_subplot(gs[0, 1])
        x = np.arange(len(layers))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, n_pos, width, label='Positive', 
                       color='red', alpha=0.7, edgecolor='black', linewidth=1.5)
        bars2 = ax2.bar(x + width/2, n_neg, width, label='Negative', 
                       color='blue', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax2.set_ylabel('Number of Cells', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Cortical Layer', fontsize=14, fontweight='bold')
        ax2.set_title('B. Modulation Direction by Layer', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(layers)
        ax2.legend(fontsize=12, framealpha=0.9)
        ax2.grid(axis='y', alpha=0.3)
        
        # Panel C: Mean MI by layer
        ax3 = fig.add_subplot(gs[0, 2])
        bars = ax3.bar(layers, mean_MIs, color=colors[:len(layers)], alpha=0.8, 
                      edgecolor='black', linewidth=2)
        ax3.axhline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.5)
        ax3.set_ylabel('Mean Modulation Index', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Cortical Layer', fontsize=14, fontweight='bold')
        ax3.set_title('C. Modulation Strength by Layer', fontsize=14, fontweight='bold')
        
        # Set y-limits to show full range
        y_min = min(mean_MIs) * 1.2 if min(mean_MIs) < 0 else min(mean_MIs) * 0.8
        y_max = max(mean_MIs) * 1.2 if max(mean_MIs) > 0 else max(mean_MIs) * 0.8
        ax3.set_ylim([y_min, y_max])
        ax3.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for bar, val in zip(bars, mean_MIs):
            height = bar.get_height()
            if height < 0:
                va = 'top'
                y_pos = height
            else:
                va = 'bottom'
                y_pos = height
            ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{val:.3f}', ha='center', va=va, fontweight='bold')
        
        plt.suptitle('Speed Modulation Analysis - Layer Gradient', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        if save_dir:
            fig.savefig(os.path.join(save_dir, 'Figure1_LayerGradient.png'), 
                       dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(save_dir, 'Figure1_LayerGradient.pdf'), 
                       bbox_inches='tight')
            print(f"✓ Figure 1 saved to {save_dir}")
        
        return fig
    
    @staticmethod
    def _plot_detailed_analysis(overall_results, layer_results, save_dir=None):
        """
        Plot detailed analysis including distributions and statistics.
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel A: Overall MI distribution
        ax1 = fig.add_subplot(gs[0, 0])
        modulation_indices = overall_results['modulation_indices']
        
        ax1.hist(modulation_indices, bins=50, alpha=0.7, edgecolor='black', color='gray')
        ax1.axvline(0, color='black', linestyle='-', linewidth=2)
        ax1.axvline(-0.1, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax1.axvline(0.1, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Modulation Index', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Cells', fontsize=12, fontweight='bold')
        ax1.set_title('A. Distribution of Modulation Indices', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        ax1.text(0.02, 0.98, 
                f"n = {overall_results['n_total']}\n"
                f"Modulated: {overall_results['n_modulated']} ({overall_results['prop_modulated']*100:.1f}%)\n"
                f"Mean: {overall_results['mean_mod_index']:.3f}\n"
                f"Median: {overall_results['median_mod_index']:.3f}",
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Panel B: MI by layer (violin plot)
        ax2 = fig.add_subplot(gs[0, 1])
        
        layer_names = []
        mi_by_layer = []
        colors_list = ['#4472C4', '#ED7D31', '#70AD47', '#C5504B']
        
        for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
            if layer_name in layer_results and len(layer_results[layer_name]['mod_indices']) > 0:
                layer_names.append(layer_name)
                mi_by_layer.append(layer_results[layer_name]['mod_indices'])
        
        if len(mi_by_layer) > 0:
            positions = np.arange(len(layer_names))
            parts = ax2.violinplot(mi_by_layer, positions=positions,
                                   showmeans=True, showmedians=True, showextrema=True)
            
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors_list[i])
                pc.set_alpha(0.7)
            
            ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax2.set_xticks(positions)
            ax2.set_xticklabels(layer_names)
            ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Modulation Index', fontsize=12, fontweight='bold')
            ax2.set_title('B. MI Distribution by Layer', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # Panel C: P-value distribution
        ax3 = fig.add_subplot(gs[0, 2])
        
        p_values = overall_results['p_values']
        ax3.hist(p_values, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
        ax3.axvline(0.05, color='red', linestyle='--', linewidth=2, label='p = 0.05')
        ax3.set_xlabel('P-value', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Cells', fontsize=12, fontweight='bold')
        ax3.set_title('C. P-value Distribution', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Panel D: Proportion modulated comparison
        ax4 = fig.add_subplot(gs[1, 0])
        
        layers_for_plot = []
        props_for_plot = []
        
        for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
            if layer_name in layer_results:
                layers_for_plot.append(layer_name)
                props_for_plot.append(layer_results[layer_name]['prop_speed_mod'] * 100)
        
        ax4.plot(layers_for_plot, props_for_plot, 'o-', linewidth=3, markersize=12, 
                color='#2E7D32', markerfacecolor='lightgreen', markeredgecolor='black', 
                markeredgewidth=2)
        ax4.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax4.set_ylabel('% Speed-Modulated', fontsize=12, fontweight='bold')
        ax4.set_title('D. Laminar Gradient', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, max(props_for_plot) * 1.2])
        
        # Panel E: Positive/Negative ratio by layer
        ax5 = fig.add_subplot(gs[1, 1])
        
        layers_ratio = []
        ratios = []
        
        for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
            if layer_name in layer_results:
                lr = layer_results[layer_name]
                if lr['n_speed_mod'] > 0:
                    layers_ratio.append(layer_name)
                    ratio = lr['n_negative'] / lr['n_speed_mod'] * 100
                    ratios.append(ratio)
        
        if len(ratios) > 0:
            bars = ax5.bar(layers_ratio, ratios, color=colors_list[:len(ratios)], 
                          alpha=0.7, edgecolor='black', linewidth=2)
            ax5.axhline(50, color='black', linestyle='--', linewidth=1.5, alpha=0.5, 
                       label='50% (balanced)')
            ax5.set_xlabel('Layer', fontsize=12, fontweight='bold')
            ax5.set_ylabel('% Negative Modulation', fontsize=12, fontweight='bold')
            ax5.set_title('E. Negative Modulation Prevalence', fontsize=13, fontweight='bold')
            ax5.set_ylim([0, 100])
            ax5.legend(fontsize=10)
            ax5.grid(True, alpha=0.3, axis='y')
            
            # Add values on bars
            for bar, val in zip(bars, ratios):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.0f}%', ha='center', va='bottom', fontweight='bold')
        
        # Panel F: Summary statistics table
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        table_data = [['Layer', 'N Total', 'N Mod', '% Mod', 'Mean MI']]
        
        for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
            if layer_name in layer_results:
                lr = layer_results[layer_name]
                row = [
                    layer_name,
                    f"{lr['n_total']}",
                    f"{lr['n_speed_mod']}",
                    f"{lr['prop_speed_mod']*100:.1f}%",
                    f"{lr['mean_mod_index']:.3f}"
                ]
                table_data.append(row)
        
        table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.15, 0.17, 0.17, 0.17, 0.17])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            color = '#D9E1F2' if i % 2 == 0 else 'white'
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor(color)
        
        ax6.set_title('F. Summary Statistics', fontsize=13, fontweight='bold', pad=20)
        
        plt.suptitle('Speed Modulation Analysis - Detailed Results', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_dir:
            fig.savefig(os.path.join(save_dir, 'Figure2_DetailedAnalysis.png'),
                        dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(save_dir, 'Figure2_DetailedAnalysis.pdf'), 
                       bbox_inches='tight')
            print(f"✓ Figure 2 saved to {save_dir}")
        
        return fig
    
    @staticmethod
    def _plot_example_cells_simple(overall_results, spike_data, speed_data, save_dir=None):
        """
        Plot example speed-modulated cells showing their activity vs speed.
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        speed_modulated_cells = overall_results['speed_modulated_cells']
        reliable_indices = overall_results['reliable_indices']
        modulation_indices = overall_results['modulation_indices']
        mean_activities = overall_results['mean_activities']
        
        if len(speed_modulated_cells) == 0:
            print("No speed-modulated cells to plot")
            return None
        
        # Get speed bins
        speed_bins = overall_results['speed_bins']
        slow_mask = (speed_data >= speed_bins['slow'][0]) & (speed_data < speed_bins['slow'][1])
        medium_mask = (speed_data >= speed_bins['medium'][0]) & (speed_data < speed_bins['medium'][1])
        fast_mask = (speed_data >= speed_bins['fast'][0]) & (speed_data < speed_bins['fast'][1])
        
        # Sort cells by absolute MI
        cell_mis = np.array([modulation_indices[list(reliable_indices).index(c)] 
                            for c in speed_modulated_cells])
        sorted_idx = np.argsort(np.abs(cell_mis))[::-1]
        
        # Plot top 12 cells (6 positive, 6 negative if possible)
        positive_cells = [speed_modulated_cells[i] for i in sorted_idx if cell_mis[sorted_idx.tolist().index(i)] > 0]
        negative_cells = [speed_modulated_cells[i] for i in sorted_idx if cell_mis[sorted_idx.tolist().index(i)] < 0]
        
        cells_to_plot = positive_cells[:6] + negative_cells[:6]
        cells_to_plot = cells_to_plot[:12]  # Max 12
        
        for plot_idx, cell_idx in enumerate(cells_to_plot):
            if plot_idx >= 12:
                break
            
            row = plot_idx // 4
            col = plot_idx % 4
            ax = fig.add_subplot(gs[row, col])
            
            # Get cell's MI and mean activities
            rel_idx = np.where(reliable_indices == cell_idx)[0][0]
            mi = modulation_indices[rel_idx]
            mean_acts = mean_activities[rel_idx]
            
            # Plot speed tuning curve
            speed_categories = ['Slow\n(2-10)', 'Medium\n(10-20)', 'Fast\n(>20)']
            colors_bar = ['blue', 'orange', 'red']
            
            bars = ax.bar(speed_categories, mean_acts, color=colors_bar, 
                         alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add connecting line
            ax.plot(range(3), mean_acts, 'k--', linewidth=2, alpha=0.5)
            
            ax.set_ylabel('Mean Activity', fontsize=10)
            # ax.set_xlabel('Speed Category', fontsize=10)
            
            # Title with cell info
            direction = "↑" if mi > 0 else "↓"
            color_title = 'red' if mi > 0 else 'blue'
            ax.set_title(f'Cell {cell_idx} {direction}\nMI = {mi:.3f}', 
                        fontsize=11, fontweight='bold', color=color_title)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Example Speed-Modulated Cells', 
                    fontsize=16, fontweight='bold')
        
        if save_dir:
            fig.savefig(os.path.join(save_dir, 'Figure3_ExampleCells.png'), 
                       dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(save_dir, 'Figure3_ExampleCells.pdf'), 
                       bbox_inches='tight')
            print(f"✓ Figure 3 saved to {save_dir}")
        
        return fig
    
    # =========================================================================
    # METHOD 2: SALEEM Q_S METHOD (FOR COMPARISON)
    # =========================================================================
    
    @staticmethod
    def create_speed_bins(min_speed=1.0, max_speed=30.0, n_bins=30):
        """
        Create speed bins following Saleem et al. method.
        
        Parameters:
        -----------
        min_speed : float
            Minimum speed for binning (speeds below this are 'stationary')
        max_speed : float
            Maximum speed for binning
        n_bins : int
            Number of speed bins (not counting stationary bin)
            
        Returns:
        --------
        speed_bins : numpy.ndarray
            Bin edges (length n_bins + 2, includes stationary bin)
        bin_centers : numpy.ndarray
            Center of each bin (length n_bins + 1)
        """
        # Create bins from min_speed to max_speed
        speed_bins = np.linspace(min_speed, max_speed, n_bins + 1)
        
        # Add stationary bin at the beginning (0 to min_speed)
        speed_bins = np.concatenate(([0, min_speed], speed_bins[1:]))
        
        # Calculate bin centers
        bin_centers = (speed_bins[:-1] + speed_bins[1:]) / 2
        
        # Set stationary bin center to 0
        bin_centers[0] = 0
        
        return speed_bins, bin_centers
    
    @staticmethod
    def extract_speed_per_lap(speed_cm_s, lap_starts, lap_ends, spike_data):
        """Extract speed and spike data for each lap separately."""
        n_laps = len(lap_starts)
        speed_laps = []
        spike_laps = []
        
        for lap_idx in range(n_laps):
            start = lap_starts[lap_idx]
            end = lap_ends[lap_idx]
            
            speed_laps.append(speed_cm_s[start:end])
            spike_laps.append(spike_data[:, start:end])
        
        return speed_laps, spike_laps
    
    @staticmethod
    def filter_running_periods(speed_lap, spike_lap, min_speed=1.0):
        """Filter to include only running periods (speed > min_speed)."""
        running_mask = speed_lap >= min_speed
        
        if np.sum(running_mask) == 0:
            return None, None
        
        speed_running = speed_lap[running_mask]
        spike_running = spike_lap[:, running_mask]
        
        return speed_running, spike_running
    
    @staticmethod
    def build_speed_response_map(spike_data, speed_data, speed_bins, 
                                 smooth_sigma=None, cv_optimize=True):
        """Build speed response map using Saleem's method."""
        n_cells = spike_data.shape[0]
        n_bins = len(speed_bins) - 1
        
        # Initialize spike-count and occupancy maps
        spike_count_map = np.zeros((n_cells, n_bins))
        occupancy_map = np.zeros(n_bins)
        
        # Build maps
        for frame_idx in range(len(speed_data)):
            speed = speed_data[frame_idx]
            
            # Find which bin this speed belongs to
            bin_idx = np.digitize(speed, speed_bins) - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            
            # Update maps
            spike_count_map[:, bin_idx] += spike_data[:, frame_idx]
            occupancy_map[bin_idx] += 1
        
        # Smooth maps (except stationary bin)
        if smooth_sigma is None and cv_optimize:
            sigma_range = np.linspace(0.5, 3.0, 10)
            best_sigma = sigma_range[len(sigma_range)//2]
            smooth_sigma = best_sigma
        elif smooth_sigma is None:
            smooth_sigma = 1.0
        
        # Apply Gaussian smoothing (preserve stationary bin)
        smoothed_spike_count = np.zeros_like(spike_count_map)
        smoothed_occupancy = np.zeros_like(occupancy_map)
        
        for cell_idx in range(n_cells):
            if n_bins > 2:
                smoothed_spike_count[cell_idx, 1:] = gaussian_filter1d(
                    spike_count_map[cell_idx, 1:], sigma=smooth_sigma
                )
                smoothed_spike_count[cell_idx, 0] = spike_count_map[cell_idx, 0]
            else:
                smoothed_spike_count[cell_idx, :] = spike_count_map[cell_idx, :]
        
        if n_bins > 2:
            smoothed_occupancy[1:] = gaussian_filter1d(occupancy_map[1:], sigma=smooth_sigma)
            smoothed_occupancy[0] = occupancy_map[0]
        else:
            smoothed_occupancy = occupancy_map
        
        # Calculate response map (firing rate)
        response_map = np.zeros((n_cells, n_bins))
        for bin_idx in range(n_bins):
            if smoothed_occupancy[bin_idx] > 0:
                response_map[:, bin_idx] = smoothed_spike_count[:, bin_idx] / smoothed_occupancy[bin_idx]
        
        return response_map, smooth_sigma
    
    @staticmethod
    def calculate_Q_S(train_response, test_spikes, test_speed, speed_bins):
        """Calculate Q_S (speed prediction quality) using cross-validation."""
        n_cells = test_spikes.shape[0]
        n_bins = len(speed_bins) - 1
        Q_S = np.zeros(n_cells)
        
        for cell_idx in range(n_cells):
            actual_rate = test_spikes[cell_idx, :]
            predicted_rate = np.zeros_like(actual_rate)
            
            for frame_idx in range(len(test_speed)):
                speed = test_speed[frame_idx]
                bin_idx = np.digitize(speed, speed_bins) - 1
                bin_idx = np.clip(bin_idx, 0, n_bins - 1)
                
                predicted_rate[frame_idx] = train_response[cell_idx, bin_idx]
            
            # Calculate prediction quality
            mean_rate = np.mean(actual_rate)
            
            ss_total = np.sum((actual_rate - mean_rate) ** 2)
            ss_residual = np.sum((actual_rate - predicted_rate) ** 2)
            
            if ss_total > 0:
                Q_S[cell_idx] = 1 - (ss_residual / ss_total)
            else:
                Q_S[cell_idx] = 0
        
        return Q_S
    
    @staticmethod
    def classify_speed_tuning(response_map, bin_centers, Q_S_threshold=0.1):
        """Classify speed tuning type following Saleem et al. (2013) Fig 2d."""
        n_cells = response_map.shape[0]
        tuning_types = np.zeros(n_cells, dtype=int)
        preferred_speeds = np.zeros(n_cells)
        
        for cell_idx in range(n_cells):
            response = response_map[cell_idx, :]
            
            # Find preferred speed (excluding stationary bin)
            if len(response) > 1:
                peak_idx = np.argmax(response[1:]) + 1
                preferred_speed = bin_centers[peak_idx]
            else:
                preferred_speed = bin_centers[0]
            
            preferred_speeds[cell_idx] = preferred_speed
            
            # Classify tuning type
            if preferred_speed <= 2.0:
                tuning_types[cell_idx] = 1  # Low-pass
            elif preferred_speed >= 25.0:
                tuning_types[cell_idx] = 3  # High-pass
            else:
                tuning_types[cell_idx] = 2  # Band-pass
        
        return tuning_types, preferred_speeds
    
    @staticmethod
    def analyze_speed_tuning_by_layer(spike_data, speed_data, lap_starts, lap_ends,
                                     layer_cells, reliable_cells, framerate,
                                     min_speed=1.0, max_speed=30.0, n_bins=30,
                                     Q_S_threshold=0.1):
        """
        Main function: Analyze speed tuning properties by cortical layer using Q_S.
        
        This is the Saleem et al. (2013) method. For most datasets, use the
        simple_speed_modulation method instead.
        """
        print("="*80)
        print("SPEED TUNING ANALYSIS BY LAYER (Q_S METHOD)")
        print("="*80)
        print(f"Method: Saleem et al. (2013) Nature Neuroscience")
        print(f"Total cells: {spike_data.shape[0]}")
        print(f"Reliable cells: {np.sum(reliable_cells)}")
        print(f"Total laps: {len(lap_starts)}")
        print(f"Speed bins: {n_bins} + 1 stationary (<{min_speed} cm/s)")
        print(f"Cross-validation: Even/odd lap split (50/50)")
        print("="*80)
        
        # Create speed bins
        speed_bins, bin_centers = SpeedTuningAnalysis.create_speed_bins(
            min_speed, max_speed, n_bins
        )
        
        # Extract speed and spikes per lap
        print("\nExtracting data per lap...")
        speed_laps, spike_laps = SpeedTuningAnalysis.extract_speed_per_lap(
            speed_data, lap_starts, lap_ends, spike_data
        )
        
        n_laps = len(speed_laps)
        print(f"Extracted {n_laps} laps")
        
        # Split into even and odd laps
        even_laps = np.arange(0, n_laps, 2)
        odd_laps = np.arange(1, n_laps, 2)
        
        print(f"Even laps (training): {len(even_laps)}")
        print(f"Odd laps (testing): {len(odd_laps)}")
        
        # Concatenate laps
        print("\nConcatenating laps for training and testing...")
        
        train_speed_list = []
        train_spike_list = []
        test_speed_list = []
        test_spike_list = []
        
        for lap_idx in even_laps:
            speed_run, spike_run = SpeedTuningAnalysis.filter_running_periods(
                speed_laps[lap_idx], spike_laps[lap_idx], min_speed
            )
            if speed_run is not None:
                train_speed_list.append(speed_run)
                train_spike_list.append(spike_run)
        
        for lap_idx in odd_laps:
            speed_run, spike_run = SpeedTuningAnalysis.filter_running_periods(
                speed_laps[lap_idx], spike_laps[lap_idx], min_speed
            )
            if speed_run is not None:
                test_speed_list.append(speed_run)
                test_spike_list.append(spike_run)
        
        train_speed = np.concatenate(train_speed_list)
        train_spikes = np.concatenate(train_spike_list, axis=1)
        test_speed = np.concatenate(test_speed_list)
        test_spikes = np.concatenate(test_spike_list, axis=1)
        
        print(f"Training frames: {len(train_speed)}")
        print(f"Testing frames: {len(test_speed)}")
        
        # Build speed response maps
        print("\nBuilding speed response maps (Saleem method)...")
        response_map, optimal_sigma = SpeedTuningAnalysis.build_speed_response_map(
            train_spikes, train_speed, speed_bins, smooth_sigma=1.0, cv_optimize=False
        )
        
        print(f"Smoothing sigma: {optimal_sigma:.2f} bins")
        
        # Calculate Q_S
        print("\nCalculating Q_S (prediction quality)...")
        Q_S = SpeedTuningAnalysis.calculate_Q_S(
            response_map, test_spikes, test_speed, speed_bins
        )
        
        # Classify tuning types
        print("\nClassifying speed tuning types...")
        tuning_types, preferred_speeds = SpeedTuningAnalysis.classify_speed_tuning(
            response_map, bin_centers, Q_S_threshold
        )
        
        # Filter for reliable cells
        Q_S_reliable = Q_S[reliable_cells]
        tuning_types_reliable = tuning_types[reliable_cells]
        preferred_speeds_reliable = preferred_speeds[reliable_cells]
        response_map_reliable = response_map[reliable_cells, :]
        
        # Count speed-tuned cells
        n_speed_tuned = np.sum(Q_S_reliable > Q_S_threshold)
        print(f"\nSpeed-tuned cells (Q_S > {Q_S_threshold}): {n_speed_tuned}/{len(Q_S_reliable)} ({n_speed_tuned/len(Q_S_reliable)*100:.1f}%)")
        
        # Analyze by layer
        print("\n" + "="*80)
        print("LAYER-SPECIFIC ANALYSIS")
        print("="*80)
        
        layer_results = {}
        layer_names = ['L2/3', 'L4', 'L5', 'L6']
        
        for layer_name in layer_names:
            if layer_name not in layer_cells:
                continue
            
            layer_indices = layer_cells[layer_name]
            reliable_layer_cells = np.intersect1d(
                np.where(reliable_cells)[0], layer_indices
            )
            
            if len(reliable_layer_cells) == 0:
                print(f"\n{layer_name}: No reliable cells")
                continue
            
            # Extract data for this layer
            layer_Q_S = Q_S[reliable_layer_cells]
            layer_tuning = tuning_types[reliable_layer_cells]
            layer_pref_speed = preferred_speeds[reliable_layer_cells]
            layer_response = response_map[reliable_layer_cells, :]
            
            # Calculate statistics
            n_tuned = np.sum(layer_Q_S > Q_S_threshold)
            prop_tuned = n_tuned / len(layer_Q_S)
            
            mean_Q_S = np.mean(layer_Q_S)
            median_Q_S = np.median(layer_Q_S)
            
            # Count tuning types
            tuned_mask = layer_Q_S > Q_S_threshold
            if np.sum(tuned_mask) > 0:
                tuned_types = layer_tuning[tuned_mask]
                n_low = np.sum(tuned_types == 1)
                n_band = np.sum(tuned_types == 2)
                n_high = np.sum(tuned_types == 3)
            else:
                n_low = n_band = n_high = 0
            
            layer_results[layer_name] = {
                'cell_indices': reliable_layer_cells,
                'Q_S': layer_Q_S,
                'tuning_types': layer_tuning,
                'preferred_speeds': layer_pref_speed,
                'response_maps': layer_response,
                'n_cells': len(reliable_layer_cells),
                'n_tuned': n_tuned,
                'prop_tuned': prop_tuned,
                'mean_Q_S': mean_Q_S,
                'median_Q_S': median_Q_S,
                'n_low_pass': n_low,
                'n_band_pass': n_band,
                'n_high_pass': n_high
            }
            
            print(f"\n{layer_name}:")
            print(f"  Reliable cells: {len(reliable_layer_cells)}")
            print(f"  Speed-tuned: {n_tuned} ({prop_tuned*100:.1f}%)")
            print(f"  Mean Q_S: {mean_Q_S:.3f}")
            print(f"  Median Q_S: {median_Q_S:.3f}")
            print(f"  Tuning types (among tuned):")
            print(f"    Low-pass: {n_low}")
            print(f"    Band-pass: {n_band}")
            print(f"    High-pass: {n_high}")
        
        # Statistical comparisons
        print("\n" + "="*80)
        print("STATISTICAL COMPARISONS")
        print("="*80)
        
        SpeedTuningAnalysis._print_statistical_comparisons(layer_results, layer_names)
        
        # Compile results
        results = {
            'speed_bins': speed_bins,
            'bin_centers': bin_centers,
            'response_map': response_map,
            'Q_S': Q_S,
            'tuning_types': tuning_types,
            'preferred_speeds': preferred_speeds,
            'reliable_cells': reliable_cells,
            'layer_results': layer_results,
            'Q_S_threshold': Q_S_threshold,
            'n_speed_tuned_total': n_speed_tuned,
            'analysis_params': {
                'min_speed': min_speed,
                'max_speed': max_speed,
                'n_bins': n_bins,
                'framerate': framerate,
                'optimal_sigma': optimal_sigma
            }
        }
        
        return results
    
    @staticmethod
    def _print_statistical_comparisons(layer_results, layer_names):
        """Print statistical comparisons between layers."""
        
        # Chi-square test for proportion tuned
        print("\n1. Proportion of speed-tuned cells:")
        
        observed = []
        layer_names_valid = []
        
        for layer_name in layer_names:
            if layer_name in layer_results:
                lr = layer_results[layer_name]
                observed.append([lr['n_tuned'], lr['n_cells'] - lr['n_tuned']])
                layer_names_valid.append(layer_name)
        
        if len(observed) >= 2:
            observed = np.array(observed)
            try:
                chi2, p_value = stats.chi2_contingency(observed)[:2]
                print(f"   Chi-square test: χ² = {chi2:.3f}, p = {p_value:.4f}")
                if p_value < 0.05:
                    print("   *** Significant difference in proportion tuned across layers")
                else:
                    print("   No significant difference in proportion tuned")
            except:
                print("   Chi-square test could not be performed")
        
        # Kruskal-Wallis test
        print("\n2. Distribution of Q_S values:")
        
        Q_S_by_layer = []
        for layer_name in layer_names_valid:
            Q_S_by_layer.append(layer_results[layer_name]['Q_S'])
        
        if len(Q_S_by_layer) >= 2:
            try:
                h_stat, p_value = stats.kruskal(*Q_S_by_layer)
                print(f"   Kruskal-Wallis test: H = {h_stat:.3f}, p = {p_value:.4f}")
                if p_value < 0.05:
                    print("   *** Significant difference in Q_S across layers")
                else:
                    print("   No significant difference in Q_S")
            except:
                print("   Kruskal-Wallis test could not be performed")
        
        # Pairwise comparisons
        if len(layer_names_valid) >= 2:
            print("\n3. Pairwise comparisons (Mann-Whitney U):")
            for i, layer1 in enumerate(layer_names_valid):
                for layer2 in layer_names_valid[i+1:]:
                    Q_S_1 = layer_results[layer1]['Q_S']
                    Q_S_2 = layer_results[layer2]['Q_S']
                    
                    try:
                        u_stat, p_value = stats.mannwhitneyu(Q_S_1, Q_S_2, alternative='two-sided')
                        sig_marker = "***" if p_value < 0.05 else ""
                        print(f"   {layer1} vs {layer2}: U = {u_stat:.1f}, p = {p_value:.4f} {sig_marker}")
                    except:
                        print(f"   {layer1} vs {layer2}: Could not perform test")
    
    # =========================================================================
    # METHOD 3: COMPARISON FUNCTION
    # =========================================================================
    
    @staticmethod
    def compare_methods(spike_data, speed_data, lap_starts, lap_ends,
                       layer_cells, reliable_cells, framerate):
        """
        Compare Simple vs Q_S methods on the same dataset.
        
        Parameters:
        -----------
        spike_data : numpy.ndarray
            Neural activity
        speed_data : numpy.ndarray
            Speed data
        lap_starts, lap_ends : numpy.ndarray
            Lap boundaries
        layer_cells : dict
            Layer assignments
        reliable_cells : numpy.ndarray
            Reliable cell mask
        framerate : float
            Recording framerate
            
        Returns:
        --------
        comparison : dict
            Results from both methods with comparison statistics
        """
        print("\n" + "="*80)
        print("COMPARING SPEED MODULATION METHODS")
        print("="*80)
        
        # Method 1: Simple
        print("\n### RUNNING SIMPLE METHOD ###\n")
        simple_results = SpeedTuningAnalysis.analyze_simple_by_layer(
            spike_data, speed_data, layer_cells, reliable_cells
        )
        
        # Method 2: Q_S
        print("\n### RUNNING Q_S METHOD ###\n")
        QS_results = SpeedTuningAnalysis.analyze_speed_tuning_by_layer(
            spike_data, speed_data, lap_starts, lap_ends,
            layer_cells, reliable_cells, framerate,
            Q_S_threshold=0.05
        )
        
        # Compare
        print("\n" + "="*80)
        print("METHOD COMPARISON")
        print("="*80)
        
        if simple_results is not None:
            simple_n = simple_results['overall']['n_modulated']
            simple_pct = simple_results['overall']['prop_modulated'] * 100
        else:
            simple_n = 0
            simple_pct = 0
        
        QS_n = QS_results['n_speed_tuned_total']
        QS_pct = QS_n / np.sum(reliable_cells) * 100
        
        print(f"\nSimple Method:")
        print(f"  Speed-modulated cells: {simple_n} ({simple_pct:.1f}%)")
        
        print(f"\nQ_S Method:")
        print(f"  Speed-tuned cells: {QS_n} ({QS_pct:.1f}%)")
        
        print(f"\nDifference:")
        print(f"  Simple found {simple_n - QS_n} more cells ({simple_pct - QS_pct:.1f}% more)")
        
        # Layer-by-layer comparison
        print(f"\n{'Layer':<8} {'Simple %':<12} {'Q_S %':<12} {'Difference':<12}")
        print("-" * 50)
        
        for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
            simple_pct_layer = 0
            QS_pct_layer = 0
            
            if simple_results and layer_name in simple_results['layer_results']:
                simple_pct_layer = simple_results['layer_results'][layer_name]['prop_speed_mod'] * 100
            
            if layer_name in QS_results['layer_results']:
                QS_pct_layer = QS_results['layer_results'][layer_name]['prop_tuned'] * 100
            
            diff = simple_pct_layer - QS_pct_layer
            print(f"{layer_name:<8} {simple_pct_layer:>10.1f}% {QS_pct_layer:>10.1f}% {diff:>+10.1f}%")
        
        print("\n" + "="*80)
        print("RECOMMENDATION:")
        if simple_n > QS_n * 2:
            print("✓ Use SIMPLE method - Q_S is too conservative for your data")
            print("  Reasons: Fewer trials, trial variability, negative modulation")
        elif simple_n > QS_n:
            print("✓ Use SIMPLE method - finds more cells while maintaining significance")
        else:
            print("✓ Both methods agree - either is appropriate")
        print("="*80)
        
        return {
            'simple': simple_results,
            'QS': QS_results,
            'comparison': {
                'simple_n': simple_n,
                'simple_pct': simple_pct,
                'QS_n': QS_n,
                'QS_pct': QS_pct,
                'difference_n': simple_n - QS_n,
                'difference_pct': simple_pct - QS_pct
            }
        }


# Example usage and testing
if __name__ == "__main__":
    print("SpeedTuningAnalysis module loaded successfully")
    print("\nAvailable methods:")
    print("1. simple_speed_modulation() - RECOMMENDED")
    print("2. analyze_simple_by_layer() - Layer-specific simple analysis")
    print("3. plot_simple_results() - Publication figures")
    print("4. analyze_speed_tuning_by_layer() - Q_S method (for comparison)")
    print("5. compare_methods() - Compare both methods")