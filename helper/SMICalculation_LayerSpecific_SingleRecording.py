"""
SMICalculation_LayerSpecific_Singlerecording.py
Refactored so the full SMI workflow is callable as Run_SMI_Layer_Analysis(...)

Calculates Spatial Modulation Index (SMI) for specific layers in mouse V1
for a single recording session, with onset filtering matching landmark analysis.

JSY, 2025 (Refactor)
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import h5py
import traceback

from helper import files, TwoP
from helper import SMI_Calculation as SMI
from helper.SpatialModulationIndexLayerSpecific import SpatialModulationIndexLayerSpecific as SMI_Layer


def filter_onset_response_cells(spatial_activity, bin_centers,
                                reliable_cells=None,
                                exclude_first_bins=10,
                                exclude_last_bins=10,
                                verbose=True):
    """
    Identify and filter out cells with peak response in onset/reward regions.
    Matches the filtering approach used in landmark preference analysis.
    """
    n_cells = spatial_activity.shape[0]

    min_pos = np.min(bin_centers)
    max_pos = np.max(bin_centers)
    bin_spacing = np.mean(np.diff(bin_centers))

    onset_threshold = min_pos + (exclude_first_bins * bin_spacing)
    end_threshold = max_pos - (exclude_last_bins * bin_spacing)

    non_onset_cells = np.ones(n_cells, dtype=bool)
    rejected_onset, rejected_reward, rejected_zero = [], [], []

    mean_profiles = np.mean(spatial_activity, axis=1)
    cells_to_check = np.where(reliable_cells)[0] if reliable_cells is not None else np.arange(n_cells)
    peak_positions = np.zeros(n_cells)

    for cell_idx in cells_to_check:
        profile = mean_profiles[cell_idx]
        global_peak_idx = np.argmax(profile)
        global_peak_pos = bin_centers[global_peak_idx]
        peak_positions[cell_idx] = global_peak_pos

        if profile[global_peak_idx] == 0:
            rejected_zero.append(cell_idx)
            non_onset_cells[cell_idx] = False
        elif global_peak_pos < onset_threshold:
            rejected_onset.append(cell_idx)
            non_onset_cells[cell_idx] = False
        elif global_peak_pos > end_threshold:
            rejected_reward.append(cell_idx)
            non_onset_cells[cell_idx] = False

    rejected_info = {
        'onset': np.array(rejected_onset), 'reward': np.array(rejected_reward),
        'zero_activity': np.array(rejected_zero), 'peak_positions': peak_positions,
        'onset_threshold': onset_threshold, 'end_threshold': end_threshold,
    }

    if verbose:
        n_rejected = len(rejected_onset) + len(rejected_reward) + len(rejected_zero)
        print(f"\nOnset/Reward Filtering: Onset < {onset_threshold:.1f}cm, Reward > {end_threshold:.1f}cm")
        print(f"  Rejected: {n_rejected} (Onset: {len(rejected_onset)}, Reward: {len(rejected_reward)}, Zero: {len(rejected_zero)})")

    return non_onset_cells, rejected_info


def save_smi_results(save_path, session_id, date_str, animal_id,
                     smi_results, layer_results, layer_cells, layer_boundaries,
                     analysis_reliable_cells, med_coords, bin_centers,
                     rejected_info, parameters):
    """Save SMI analysis results to HDF5 file for downstream analysis."""
    print(f"\n=== SAVING SMI RESULTS ===")
    print(f"Output: {os.path.basename(save_path)}")

    def sanitize_name(name):
        return str(name).replace('/', '_').replace('\\', '_')

    with h5py.File(save_path, 'w') as f:
        # Metadata
        f.attrs['session_id'] = session_id
        f.attrs['date'] = date_str
        f.attrs['animal_id'] = animal_id
        f.attrs['n_cells_total'] = len(analysis_reliable_cells)
        f.attrs['n_cells_analyzed'] = int(np.sum(analysis_reliable_cells))

        # Parameters
        params_grp = f.create_group('parameters')
        for key, value in parameters.items():
            if value is not None and not isinstance(value, (list, np.ndarray)):
                params_grp.attrs[key] = value

        # Global SMI results
        global_grp = f.create_group('global_smi')
        SMI_values = smi_results['smi_results']['SMI']
        valid_cells = smi_results['smi_results']['reliable_valid_cells']

        global_grp.create_dataset('SMI_all_cells', data=SMI_values)
        global_grp.create_dataset('valid_cells_mask', data=valid_cells)
        global_grp.create_dataset('analysis_reliable_cells', data=analysis_reliable_cells)
        global_grp.create_dataset('preferred_positions', data=smi_results['smi_results']['preferred_positions'])
        global_grp.create_dataset('non_preferred_positions', data=smi_results['smi_results']['non_preferred_positions'])
        global_grp.create_dataset('Rp', data=smi_results['smi_results']['Rp'])
        global_grp.create_dataset('Rn', data=smi_results['smi_results']['Rn'])

        SMI_clean = SMI_values[valid_cells]
        SMI_clean = SMI_clean[~np.isnan(SMI_clean) & ~np.isinf(SMI_clean)]
        global_grp.attrs['n_valid_cells'] = len(SMI_clean)
        global_grp.attrs['median_smi'] = float(np.median(SMI_clean)) if len(SMI_clean) > 0 else np.nan
        global_grp.attrs['mad_smi'] = float(stats.median_abs_deviation(SMI_clean)) if len(SMI_clean) > 0 else np.nan
        global_grp.attrs['mean_smi'] = float(np.mean(SMI_clean)) if len(SMI_clean) > 0 else np.nan
        global_grp.attrs['std_smi'] = float(np.std(SMI_clean)) if len(SMI_clean) > 0 else np.nan

        # Cell coordinates and layer info
        coords_grp = f.create_group('cell_info')
        coords_grp.create_dataset('med_coords', data=med_coords)
        coords_grp.create_dataset('bin_centers', data=bin_centers)

        # Layer boundaries
        boundaries_grp = coords_grp.create_group('layer_boundaries')
        for layer_name, (upper, lower) in layer_boundaries.items():
            safe_name = sanitize_name(layer_name)
            boundaries_grp.attrs[f'{safe_name}_upper'] = upper
            boundaries_grp.attrs[f'{safe_name}_lower'] = lower

        # Layer-specific results
        layers_grp = f.create_group('layer_smi')

        for layer_name, layer_cell_indices in layer_cells.items():
            safe_name = sanitize_name(layer_name)
            layer_grp = layers_grp.create_group(safe_name)
            layer_grp.attrs['original_name'] = layer_name
            layer_grp.create_dataset('cell_indices', data=layer_cell_indices)

            # Get layer results if available
            if layer_results is not None and layer_name in layer_results and layer_results[layer_name] is not None:
                lr = layer_results[layer_name]
                layer_grp.create_dataset('reliable_valid_cells', data=lr['reliable_valid_cells'])
                layer_grp.create_dataset('SMI', data=lr['SMI'])

                layer_grp.attrs['n_cells_total'] = len(layer_cell_indices)
                layer_grp.attrs['n_cells_valid'] = len(lr['SMI'])
                layer_grp.attrs['median_smi'] = float(lr['stats']['median'])
                layer_grp.attrs['mean_smi'] = float(lr['stats']['mean'])
                layer_grp.attrs['std_smi'] = float(lr['stats']['std'])
                layer_grp.attrs['sem_smi'] = float(lr['stats']['sem'])

                # MAD calculation
                if len(lr['SMI']) > 0:
                    layer_grp.attrs['mad_smi'] = float(stats.median_abs_deviation(lr['SMI']))

                # Proportion of spatially modulated cells (SMI > 0.1)
                n_modulated = np.sum(lr['SMI'] > 0.1)
                layer_grp.attrs['n_modulated'] = int(n_modulated)
                layer_grp.attrs['prop_modulated'] = float(n_modulated / len(lr['SMI'])) if len(lr['SMI']) > 0 else 0.0

                if lr['preferred_positions'] is not None:
                    layer_grp.create_dataset('preferred_positions', data=lr['preferred_positions'])
                if lr['Rp'] is not None:
                    layer_grp.create_dataset('Rp', data=lr['Rp'])
                if lr['Rn'] is not None:
                    layer_grp.create_dataset('Rn', data=lr['Rn'])
            else:
                layer_grp.attrs['n_cells_total'] = len(layer_cell_indices)
                layer_grp.attrs['n_cells_valid'] = 0

        # Rejected cells info
        rejected_grp = f.create_group('rejected_cells')
        rejected_grp.create_dataset('onset', data=rejected_info['onset'])
        rejected_grp.create_dataset('reward', data=rejected_info['reward'])
        rejected_grp.create_dataset('zero_activity', data=rejected_info['zero_activity'])
        rejected_grp.attrs['onset_threshold'] = rejected_info['onset_threshold']
        rejected_grp.attrs['end_threshold'] = rejected_info['end_threshold']

    print(f"✓ Saved: {os.path.basename(save_path)}")


def visualize_onset_filtering(spatial_activity, reliable_cells, non_onset_cells,
                              onset_cells_mask, peak_positions, bin_centers, save_path=None):
    """Create visualization showing which cells were filtered as onset responses."""
    from matplotlib.colors import LinearSegmentedColormap

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    cmap = LinearSegmentedColormap.from_list('Blues', [(1,1,1), (0.4,0.4,0.9), (0,0,0.5)])

    def create_sorted_plot(activity, mask):
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return None
        act = activity[indices]
        n_trials = act.shape[1]
        avg = np.mean(act[:, np.arange(1, n_trials, 2), :], axis=1)
        sorted_idx = np.argsort(np.argmax(avg, axis=1))
        sorted_act = avg[sorted_idx]
        for i in range(len(sorted_act)):
            r = sorted_act[i]
            sorted_act[i] = (r - np.min(r)) / (np.max(r) - np.min(r) + 1e-10)
        return sorted_act

    # All reliable cells
    sorted_all = create_sorted_plot(spatial_activity, reliable_cells)
    if sorted_all is not None:
        axes[0,0].imshow(sorted_all, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    axes[0,0].set_title(f'All Reliable Cells ({np.sum(reliable_cells)})', fontweight='bold')
    axes[0,0].set_xlabel('Spatial Bin'); axes[0,0].set_ylabel('Cell #')

    # Filtered cells
    filtered = reliable_cells & non_onset_cells
    sorted_filt = create_sorted_plot(spatial_activity, filtered)
    if sorted_filt is not None:
        axes[0,1].imshow(sorted_filt, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    axes[0,1].set_title(f'After Onset Filtering ({np.sum(filtered)})', fontweight='bold')
    axes[0,1].set_xlabel('Spatial Bin'); axes[0,1].set_ylabel('Cell #')

    # Peak distribution
    onset_peaks = peak_positions[reliable_cells & onset_cells_mask]
    non_onset_peaks = peak_positions[reliable_cells & non_onset_cells]
    axes[1,0].hist(non_onset_peaks, bins=30, alpha=0.7, label=f'Spatial (n={len(non_onset_peaks)})')
    axes[1,0].hist(onset_peaks, bins=30, alpha=0.7, label=f'Onset (n={len(onset_peaks)})')
    axes[1,0].set_xlabel('Peak Position (cm)'); axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('Peak Position Distribution', fontweight='bold')
    axes[1,0].legend()

    # Summary
    axes[1,1].axis('off')
    txt = f"FILTERING SUMMARY\n\nReliable cells: {np.sum(reliable_cells)}\nOnset removed: {np.sum(onset_cells_mask & reliable_cells)}\nRetained: {np.sum(filtered)}"
    axes[1,1].text(0.1, 0.7, txt, fontsize=12, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def Run_SMI_Layer_Analysis(data_filepath,
                           exclude_first_bins=5,
                           exclude_last_bins=5,
                           segment_distance=28,
                           exclude_start_cm=15,
                           exclude_end_cm=10,
                           smoothing_sigma=1.0,
                           save_figures=True,
                           verbose=True):
    """
    Run the complete SMI layer-specific analysis for one recording folder.

    Parameters
    ----------
    data_filepath : str
        Full path to the TSeries folder (same as in your previous script).
    exclude_first_bins : int
        Number of spatial bins at the start to exclude (onset filter).
    exclude_last_bins : int
        Number of spatial bins at the end to exclude (reward filter).
    segment_distance : float
        Parameter passed to the SMI analyzer.
    exclude_start_cm, exclude_end_cm : float
        Boundary exclusion distances (cm).
    smoothing_sigma : float
        Gaussian smoothing sigma used in SMI calculation.
    save_figures : bool
        Save visualization figures (keeps current structure when True).
    verbose : bool
        Print progress messages.

    Returns
    -------
    results_dict : dict
        Contains keys: results_full, layer_results, analysis_reliable_cells,
                       parameters, med_coords, bin_centers, rejected_info, figures
    """

    try:
        if verbose:
            print("="*80)
            print("SMI LAYER-SPECIFIC ANALYSIS WITH ONSET FILTERING")
            print("="*80)
            print(f"Data: {data_filepath}")
            print(f"Onset filter: first {exclude_first_bins} bins, last {exclude_last_bins} bins")
            print("="*80 + "\n")

        # STEP 1: LOAD DATA
        if verbose:
            print("STEP 1: Loading preprocessed data...")

        preproc_files = glob.glob(os.path.join(data_filepath, "*preproc*.h5"))
        if not preproc_files:
            raise ValueError(f"No preprocessed .h5 file found in {data_filepath}")

        preproc_file = preproc_files[0]
        preproc_data = files.read_h5(preproc_file)

        spatial_activity = preproc_data['spatial_activity']
        normalized_spatial_activity = preproc_data['norm_spatial_activity']
        bin_centers = preproc_data['bin_centers']
        reliable_cells = preproc_data['combined_reliable']

        n_cells, n_trials, n_bins = spatial_activity.shape
        if verbose:
            print(f"  Data: {n_cells} cells, {n_trials} trials, {n_bins} bins")
            print(f"  Reliable cells: {np.sum(reliable_cells)}")

        # Prepare bin centers (same scaling used previously)
        shifted_centers = bin_centers - np.min(bin_centers)
        scaled_bin_centers = shifted_centers * (np.size(bin_centers) / np.max(shifted_centers))

        # STEP 2: GET LAYER INFORMATION
        if verbose:
            print("\nSTEP 2: Identifying cortical layers...")

        twoP_filename = os.path.basename(data_filepath)
        raw_twop_data = TwoP(data_filepath, twoP_filename)
        raw_twop_data.find_files()
        twop_dict = raw_twop_data.calc_dFF()

        med_coords = np.array([cell['med'] for cell in twop_dict['stat']])
        layer_cells, layer_boundaries = SMI_Layer.identify_layers(med_coords)

        # STEP 3: APPLY ONSET FILTERING
        if verbose:
            print("\nSTEP 3: Applying onset/reward filtering...")

        non_onset_cells, rejected_info = filter_onset_response_cells(
            spatial_activity, scaled_bin_centers, reliable_cells,
            exclude_first_bins=exclude_first_bins, exclude_last_bins=exclude_last_bins
        )

        analysis_reliable_cells = reliable_cells & non_onset_cells
        if verbose:
            print(f"  Final cells for analysis: {np.sum(analysis_reliable_cells)}")

        # Visualize filtering
        onset_mask = ~non_onset_cells
        viz_dir = os.path.join(data_filepath, 'SMI_Figures')
        if save_figures and not os.path.exists(viz_dir):
            os.makedirs(viz_dir, exist_ok=True)
        viz_path = os.path.join(viz_dir, 'onset_filtering_visualization.png')
        fig_onset = visualize_onset_filtering(spatial_activity, reliable_cells, non_onset_cells,
                                             onset_mask, rejected_info['peak_positions'],
                                             scaled_bin_centers, viz_path if save_figures else None)

        # STEP 4: CALCULATE SMI
        if verbose:
            print("\nSTEP 4: Calculating SMI...")

        results_full = SMI.analyze_spatial_modulation_improved(
            spatial_activity, scaled_bin_centers, analysis_reliable_cells,
            segment_distance=segment_distance, exclude_start_cm=exclude_start_cm,
            exclude_end_cm=exclude_end_cm, smoothing_sigma=smoothing_sigma,
            data_filepath=data_filepath,
        )

        SMI_values = results_full['smi_results']['SMI']
        valid_cells = results_full['smi_results']['reliable_valid_cells']
        SMI_clean = SMI_values[valid_cells]
        SMI_clean = SMI_clean[~np.isnan(SMI_clean) & ~np.isinf(SMI_clean)]

        if verbose:
            print(f"\n  Valid cells: {len(SMI_clean)}")
            print(f"  Median SMI: {np.median(SMI_clean):.3f} ± {stats.median_abs_deviation(SMI_clean):.3f}")

        # STEP 5: LAYER-SPECIFIC ANALYSIS
        if verbose:
            print("\nSTEP 5: Layer-specific analysis...")

        # Create FOV for visualization
        numCells = len(twop_dict['stat'])
        im = np.zeros((twop_dict['ops']['Ly'], twop_dict['ops']['Lx']))
        for n in range(numCells):
            ypix = twop_dict['stat'][n]['ypix'][~twop_dict['stat'][n]['overlap']]
            xpix = twop_dict['stat'][n]['xpix'][~twop_dict['stat'][n]['overlap']]
            im[ypix, xpix] = xpix

        # Plot and save layer distribution
        SMI_Layer.plot_layer_distribution(med_coords, layer_cells, analysis_reliable_cells, im)
        if save_figures:
            plt.savefig(os.path.join(viz_dir, "layer_distribution.png"), dpi=300, bbox_inches='tight')

        layer_results, _ = SMI_Layer.run_layer_SMI_analysis(
            results_full['smi_results'], valid_cells, med_coords, layer_cells,
            normalized_spatial_activity, scaled_bin_centers,
            save_path=viz_dir
        )

        # STEP 6: EXTRACT SESSION INFO AND SAVE
        if verbose:
            print("\nSTEP 6: Saving results...")

        session_folder = os.path.basename(os.path.dirname(data_filepath))
        match = re.match(r'(\d{6})_.*_(Day\d+)', session_folder)

        if match:
            date_str = match.group(1)
            session_id = match.group(2)
        else:
            date_str = "unknown"
            session_id = "unknown"

        animal_match = re.search(r'(JSY\d+)', data_filepath)
        animal_id = animal_match.group(1) if animal_match else "unknown"

        parameters = {
            'exclude_first_bins': exclude_first_bins,
            'exclude_last_bins': exclude_last_bins,
            'segment_distance': segment_distance,
            'exclude_start_cm': exclude_start_cm,
            'exclude_end_cm': exclude_end_cm,
            'smoothing_sigma': smoothing_sigma,
            'n_trials': n_trials,
            'n_bins': n_bins
        }

        h5_save_path = os.path.join(data_filepath, f"{animal_id}_{session_id}_smi_results.h5")

        save_smi_results(
            h5_save_path, session_id, date_str, animal_id,
            results_full, layer_results, layer_cells, layer_boundaries,
            analysis_reliable_cells, med_coords, scaled_bin_centers,
            rejected_info, parameters
        )

        # STEP 7: SUMMARY VISUALIZATION
        if verbose:
            print("\nSTEP 7: Creating summary plots...")

        fig = None
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            layer_order = ['L2/3', 'L4', 'L5', 'L6']
            colors = ['#1E88E5', '#FF9800', '#4CAF50', '#E53935']

            data_for_plot = []
            labels = []
            for layer_name in layer_order:
                if layer_results.get(layer_name) is not None and len(layer_results[layer_name]['SMI']) > 0:
                    data_for_plot.append(layer_results[layer_name]['SMI'])
                    labels.append(f"{layer_name}\n(n={len(layer_results[layer_name]['SMI'])})")

            if data_for_plot:
                bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors[:len(data_for_plot)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)

            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylabel('SMI')
            ax.set_title(f'{animal_id} - {session_id}: SMI by Layer')
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            if save_figures:
                plt.savefig(os.path.join(data_filepath, 'smi_by_layer_boxplot.png'), dpi=300, bbox_inches='tight')
        except Exception as e:
            print("Warning: failed to create summary boxplot:", e)
            traceback.print_exc()

        if verbose:
            print("\n" + "="*80)
            print("ANALYSIS COMPLETE!")
            print("="*80)
            print(f"\nOutputs saved to: {data_filepath}")
            print(f"  - {animal_id}_{session_id}_smi_results.h5 (for downstream analysis)")
            print(f"  - onset_filtering_visualization.png")
            print(f"  - layer_distribution.png")
            print(f"  - layer_smi_comparison.png")
            print(f"  - smi_by_layer_boxplot.png")

        # Compile return dict (keeps same variables you expect)
        results = {
            'results_full': results_full,
            'layer_results': layer_results,
            'analysis_reliable_cells': analysis_reliable_cells,
            'parameters': parameters,
            'med_coords': med_coords,
            'bin_centers': scaled_bin_centers,
            'rejected_info': rejected_info,
            'figures': {
                'onset_filter': fig_onset,
                'layer_distribution': None,   # plotted directly and saved
                'smi_by_layer_boxplot': fig
            }
        }

        return results

    except Exception as err:
        # Bubble up exception after printing (keeps old behavior)
        print("ERROR in Run_SMI_Layer_Analysis:", err)
        traceback.print_exc()
        raise



def Run_SMI_AxonalImaging_window_Analysis(data_filepath,
                           exclude_first_bins=5,
                           exclude_last_bins=5,
                           segment_distance=28,
                           exclude_start_cm=15,
                           exclude_end_cm=10,
                           smoothing_sigma=1.0,
                           save_figures=True,
                           verbose=True):
    """
    Run the complete SMI layer-specific analysis for one recording folder.

    Parameters
    ----------
    data_filepath : str
        Full path to the TSeries folder (same as in your previous script).
    exclude_first_bins : int
        Number of spatial bins at the start to exclude (onset filter).
    exclude_last_bins : int
        Number of spatial bins at the end to exclude (reward filter).
    segment_distance : float
        Parameter passed to the SMI analyzer.
    exclude_start_cm, exclude_end_cm : float
        Boundary exclusion distances (cm).
    smoothing_sigma : float
        Gaussian smoothing sigma used in SMI calculation.
    save_figures : bool
        Save visualization figures (keeps current structure when True).
    verbose : bool
        Print progress messages.

    Returns
    -------
    results_dict : dict
        Contains keys: results_full, analysis_reliable_cells,
                       parameters, bin_centers, rejected_info, figures
    """

    try:
        if verbose:
            print("="*80)
            print("SMI AXONAL IMAGING WINDOW ANALYSIS WITH ONSET FILTERING")
            print("="*80)
            print(f"Data: {data_filepath}")
            print(f"Onset filter: first {exclude_first_bins} bins, last {exclude_last_bins} bins")
            print("="*80 + "\n")

        # STEP 1: LOAD DATA
        if verbose:
            print("STEP 1: Loading preprocessed data...")

        preproc_files = glob.glob(os.path.join(data_filepath, "*preproc*.h5"))
        if not preproc_files:
            raise ValueError(f"No preprocessed .h5 file found in {data_filepath}")

        preproc_file = preproc_files[0]
        preproc_data = files.read_h5(preproc_file)

        spatial_activity = preproc_data['spatial_activity']
        normalized_spatial_activity = preproc_data['norm_spatial_activity']
        bin_centers = preproc_data['bin_centers']
        reliable_cells = preproc_data['combined_reliable']

        n_cells, n_trials, n_bins = spatial_activity.shape
        if verbose:
            print(f"  Data: {n_cells} cells, {n_trials} trials, {n_bins} bins")
            print(f"  Reliable cells: {np.sum(reliable_cells)}")

        # Prepare bin centers (same scaling used previously)
        shifted_centers = bin_centers - np.min(bin_centers)
        scaled_bin_centers = shifted_centers * (np.size(bin_centers) / np.max(shifted_centers))

        # # STEP 2: GET LAYER INFORMATION
        # if verbose:
        #     print("\nSTEP 2: Identifying cortical layers...")

        # twoP_filename = os.path.basename(data_filepath)
        # raw_twop_data = TwoP(data_filepath, twoP_filename)
        # raw_twop_data.find_files()
        # twop_dict = raw_twop_data.calc_dFF()

        # med_coords = np.array([cell['med'] for cell in twop_dict['stat']])
        # layer_cells, layer_boundaries = SMI_Layer.identify_layers(med_coords)

        # STEP 3: APPLY ONSET FILTERING
        if verbose:
            print("\nSTEP 3: Applying onset/reward filtering...")

        non_onset_cells, rejected_info = filter_onset_response_cells(
            spatial_activity, scaled_bin_centers, reliable_cells,
            exclude_first_bins=exclude_first_bins, exclude_last_bins=exclude_last_bins
        )

        analysis_reliable_cells = reliable_cells & non_onset_cells
        if verbose:
            print(f"  Final cells for analysis: {np.sum(analysis_reliable_cells)}")

        # Visualize filtering
        onset_mask = ~non_onset_cells
        viz_dir = os.path.join(data_filepath, 'SMI_Figures')
        if save_figures and not os.path.exists(viz_dir):
            os.makedirs(viz_dir, exist_ok=True)
        viz_path = os.path.join(viz_dir, 'onset_filtering_visualization.png')
        fig_onset = visualize_onset_filtering(spatial_activity, reliable_cells, non_onset_cells,
                                             onset_mask, rejected_info['peak_positions'],
                                             scaled_bin_centers, viz_path if save_figures else None)

        # STEP 4: CALCULATE SMI
        if verbose:
            print("\nSTEP 4: Calculating SMI...")

        results_full = SMI.analyze_spatial_modulation_improved(
            spatial_activity, scaled_bin_centers, analysis_reliable_cells,
            segment_distance=segment_distance, exclude_start_cm=exclude_start_cm,
            exclude_end_cm=exclude_end_cm, smoothing_sigma=smoothing_sigma,
            data_filepath=data_filepath,
        )

        SMI_values = results_full['smi_results']['SMI']
        valid_cells = results_full['smi_results']['reliable_valid_cells']
        SMI_clean = SMI_values[valid_cells]
        SMI_clean = SMI_clean[~np.isnan(SMI_clean) & ~np.isinf(SMI_clean)]

        if verbose:
            print(f"\n  Valid cells: {len(SMI_clean)}")
            print(f"  Median SMI: {np.median(SMI_clean):.3f} ± {stats.median_abs_deviation(SMI_clean):.3f}")

        # # STEP 5: LAYER-SPECIFIC ANALYSIS
        # if verbose:
        #     print("\nSTEP 5: Layer-specific analysis...")

        # # Create FOV for visualization
        # numCells = len(twop_dict['stat'])
        # im = np.zeros((twop_dict['ops']['Ly'], twop_dict['ops']['Lx']))
        # for n in range(numCells):
        #     ypix = twop_dict['stat'][n]['ypix'][~twop_dict['stat'][n]['overlap']]
        #     xpix = twop_dict['stat'][n]['xpix'][~twop_dict['stat'][n]['overlap']]
        #     im[ypix, xpix] = xpix

        # # Plot and save layer distribution
        # SMI_Layer.plot_layer_distribution(med_coords, layer_cells, analysis_reliable_cells, im)
        # if save_figures:
        #     plt.savefig(os.path.join(viz_dir, "layer_distribution.png"), dpi=300, bbox_inches='tight')

        # layer_results, _ = SMI_Layer.run_layer_SMI_analysis(
        #     results_full['smi_results'], valid_cells, med_coords, layer_cells,
        #     normalized_spatial_activity, scaled_bin_centers,
        #     save_path=viz_dir
        # )

        # STEP 6: EXTRACT SESSION INFO AND SAVE
        if verbose:
            print("\nSTEP 6: Saving results...")

        session_folder = os.path.basename(os.path.dirname(data_filepath))
        match = re.match(r'(\d{6})_.*_(Day\d+)', session_folder)

        if match:
            date_str = match.group(1)
            session_id = match.group(2)
        else:
            date_str = "unknown"
            session_id = "unknown"

        animal_match = re.search(r'(JSY\d+)', data_filepath)
        animal_id = animal_match.group(1) if animal_match else "unknown"

        parameters = {
            'exclude_first_bins': exclude_first_bins,
            'exclude_last_bins': exclude_last_bins,
            'segment_distance': segment_distance,
            'exclude_start_cm': exclude_start_cm,
            'exclude_end_cm': exclude_end_cm,
            'smoothing_sigma': smoothing_sigma,
            'n_trials': n_trials,
            'n_bins': n_bins
        }

        h5_save_path = os.path.join(data_filepath, f"{animal_id}_{session_id}_smi_results.h5")

        save_smi_results_axonalimaging_window(
            h5_save_path, session_id, date_str, animal_id,
            results_full,analysis_reliable_cells, scaled_bin_centers,
            rejected_info, parameters
        )

        # # STEP 7: SUMMARY VISUALIZATION
        # if verbose:
        #     print("\nSTEP 7: Creating summary plots...")

        # fig = None
        # try:
        #     fig, ax = plt.subplots(figsize=(10, 6))
        #     layer_order = ['L2/3', 'L4', 'L5', 'L6']
        #     colors = ['#1E88E5', '#FF9800', '#4CAF50', '#E53935']

        #     data_for_plot = []
        #     labels = []
        #     for layer_name in layer_order:
        #         if layer_results.get(layer_name) is not None and len(layer_results[layer_name]['SMI']) > 0:
        #             data_for_plot.append(layer_results[layer_name]['SMI'])
        #             labels.append(f"{layer_name}\n(n={len(layer_results[layer_name]['SMI'])})")

        #     if data_for_plot:
        #         bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
        #         for patch, color in zip(bp['boxes'], colors[:len(data_for_plot)]):
        #             patch.set_facecolor(color)
        #             patch.set_alpha(0.6)

        #     ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        #     ax.set_ylabel('SMI')
        #     ax.set_title(f'{animal_id} - {session_id}: SMI by Layer')
        #     ax.grid(True, alpha=0.3, axis='y')

        #     plt.tight_layout()
        #     if save_figures:
        #         plt.savefig(os.path.join(data_filepath, 'smi_by_layer_boxplot.png'), dpi=300, bbox_inches='tight')
        # except Exception as e:
        #     print("Warning: failed to create summary boxplot:", e)
        #     traceback.print_exc()

        # if verbose:
        #     print("\n" + "="*80)
        #     print("ANALYSIS COMPLETE!")
        #     print("="*80)
        #     print(f"\nOutputs saved to: {data_filepath}")
        #     print(f"  - {animal_id}_{session_id}_smi_results.h5 (for downstream analysis)")
        #     print(f"  - onset_filtering_visualization.png")
        #     print(f"  - layer_distribution.png")
        #     print(f"  - layer_smi_comparison.png")
        #     print(f"  - smi_by_layer_boxplot.png")

        # Compile return dict (keeps same variables you expect)
        results = {
            'results_full': results_full,
            # 'layer_results': layer_results,
            'analysis_reliable_cells': analysis_reliable_cells,
            'parameters': parameters,
            # 'med_coords': med_coords,
            'bin_centers': scaled_bin_centers,
            'rejected_info': rejected_info,
            'figures': {
                'onset_filter': fig_onset
            }
        }

        return results

    except Exception as err:
        print("ERROR in Run_SMI_AxonalImaging_Window_Analysis:", err)
        traceback.print_exc()
        raise



def save_smi_results_axonalimaging_window(save_path, session_id, date_str, animal_id,
                     smi_results, analysis_reliable_cells, bin_centers,
                     rejected_info, parameters):
    """Save SMI analysis results to HDF5 file for downstream analysis."""
    print(f"\n=== SAVING SMI RESULTS ===")
    print(f"Output: {os.path.basename(save_path)}")

    def sanitize_name(name):
        return str(name).replace('/', '_').replace('\\', '_')

    with h5py.File(save_path, 'w') as f:
        # Metadata
        f.attrs['session_id'] = session_id
        f.attrs['date'] = date_str
        f.attrs['animal_id'] = animal_id
        f.attrs['n_cells_total'] = len(analysis_reliable_cells)
        f.attrs['n_cells_analyzed'] = int(np.sum(analysis_reliable_cells))

        # Parameters
        params_grp = f.create_group('parameters')
        for key, value in parameters.items():
            if value is not None and not isinstance(value, (list, np.ndarray)):
                params_grp.attrs[key] = value

        # Global SMI results
        global_grp = f.create_group('global_smi')
        SMI_values = smi_results['smi_results']['SMI']
        valid_cells = smi_results['smi_results']['reliable_valid_cells']

        global_grp.create_dataset('SMI_all_cells', data=SMI_values)
        global_grp.create_dataset('valid_cells_mask', data=valid_cells)
        global_grp.create_dataset('analysis_reliable_cells', data=analysis_reliable_cells)
        global_grp.create_dataset('preferred_positions', data=smi_results['smi_results']['preferred_positions'])
        global_grp.create_dataset('non_preferred_positions', data=smi_results['smi_results']['non_preferred_positions'])
        global_grp.create_dataset('Rp', data=smi_results['smi_results']['Rp'])
        global_grp.create_dataset('Rn', data=smi_results['smi_results']['Rn'])

        SMI_clean = SMI_values[valid_cells]
        SMI_clean = SMI_clean[~np.isnan(SMI_clean) & ~np.isinf(SMI_clean)]
        global_grp.attrs['n_valid_cells'] = len(SMI_clean)
        global_grp.attrs['median_smi'] = float(np.median(SMI_clean)) if len(SMI_clean) > 0 else np.nan
        global_grp.attrs['mad_smi'] = float(stats.median_abs_deviation(SMI_clean)) if len(SMI_clean) > 0 else np.nan
        global_grp.attrs['mean_smi'] = float(np.mean(SMI_clean)) if len(SMI_clean) > 0 else np.nan
        global_grp.attrs['std_smi'] = float(np.std(SMI_clean)) if len(SMI_clean) > 0 else np.nan


        # Rejected cells info
        rejected_grp = f.create_group('rejected_cells')
        rejected_grp.create_dataset('onset', data=rejected_info['onset'])
        rejected_grp.create_dataset('reward', data=rejected_info['reward'])
        rejected_grp.create_dataset('zero_activity', data=rejected_info['zero_activity'])
        rejected_grp.attrs['onset_threshold'] = rejected_info['onset_threshold']
        rejected_grp.attrs['end_threshold'] = rejected_info['end_threshold']

    print(f"✓ Saved: {os.path.basename(save_path)}")

