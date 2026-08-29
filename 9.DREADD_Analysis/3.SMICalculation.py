"""
Phase 3 -- SMI Calculation + Reliability (DREADD saline/DCZ cohort)
====================================================================

Consolidated from 3.SMICalculation.ipynb once every function below worked
end-to-end on real JSY090 data.

Scope
-----
Whole-session SMI computation, split into two independent tracks that must
never be merged:

- **Track B** (population-level, no cross-session tracking) -- run SMI on
  each session independently, using every cell that passes QC in that
  session. This is the track being built out first, across Phases 3-7, so
  the whole populational-level analysis exists before layering in the
  cross-session tracked-cell (Track A) work on top of it.
- **Track A** (tracked-cell, cross-session) -- joins a Phase 2 tracking
  group's `master_df` onto each session's already-computed Track B SMI
  output for a genuine within-cell paired comparison, at the cost of much
  smaller N (only cells successfully tracked across the whole group). No
  new SMI computation is involved -- SMI is a per-session, per-cell
  quantity Track B already computed and saved; Track A only joins it onto
  tracked-cell identities (`join_smi_to_master_table`, `load_track_a_group`,
  `save_track_a_joined_table`). Active as of JSY093's DCZ1/DCZ2/DCZ3
  tracking groups (saline-vs-DCZ only, no baseline anchor) -- see
  `__main__` below. Track A has NOT yet been added to Phases 4-6, which
  still only consume Track B's population-level output; that remains the
  next body of work once this joined table has been used to sanity-check
  the paired comparison here.

  **IMPORTANT, re-read before building Track A's saline/DCZ comparisons**
  (discovered while building Phase 4's Function 4.5 calibration check):
  Track B has no cross-session cell identity, so a Track B comparison can
  never restrict to "cells reliable in both sessions" using each session's
  own full data -- every Track B comparison re-establishes "reliable"
  independently per session, at that session's own trial count. Saline
  recordings are shorter than DCZ recordings within each pair, so this can
  bias BOTH the reliable-cell fraction and the SMI magnitude among
  survivors (shorter/noisier sessions show survivorship bias toward only
  the most robustly-tuned cells) -- not just a power/N issue, a real
  confound. Track A's paired within-cell comparison sidesteps this
  entirely, since the same tracked cell's SMI is compared across sessions
  regardless of either session's trial count -- it does NOT need
  trial-matching to be valid, unlike Track B. This is a real reason to
  treat Track A as the more decisive test of the saline-vs-DCZ question,
  not just a "more sensitive" alternative to Track B's population-level
  comparison. See `4.SessionComparison.ipynb`'s Function 4.5 for the
  calibration check that surfaced this.

Every Track-B function below is designed so Track A's later pass can reuse
its outputs untouched -- in particular, `run_smi_analysis_session`'s saved
`*_smi_results_dreadd.h5` is exactly what `join_smi_to_master_table` reads
via its `smi_results_by_label` argument (just keep the in-memory `result`
dict, or reload the .h5 the same lightweight way the notebook's Function
3.4 test cell did).

**DCZ pharmacokinetic time-binning (originally planned as 3b) is
deliberately deferred** -- this pass uses full-session SMI only, not
sub-windowed by time-since-injection.

Decisions locked in
--------------------
- Reliability mask for SMI: `combined_reliable` (from `preproc.h5`) -- same
  convention `Run_SMI_Layer_Analysis` already uses for every other cohort,
  and its extra peak-location-stability check is directly relevant to
  SMI's validity.
- Split-half reliability needs no new computation -- `preproc.h5`'s
  existing `avg_cc` field already *is* the real odd/even correlation
  (traced through `test_cell_reliability_improved`'s shuffle loop to
  confirm this).
- New SMI output uses a distinct filename (`*_smi_results_dreadd.h5`) so it
  never collides with any pre-existing `smi_results.h5` for these sessions.
- Reliability-failure diagnostic (Functions 3.3/3.3b + the Phase-0
  sensitivity check below) found `pattern_unstable` (odd/even peak-location
  instability) is the dominant failure mode for this cohort (94.6% of
  failures), NOT the CC floor or Cohen's d. A shuffle-normalized
  alternative to the flat `peak_distance_threshold` was tested and made
  things WORSE (299->235 combined_reliable on the Day1 test session) --
  conclusion: keep the existing `combined_reliable` criterion as-is. This
  remains an open methods question worth revisiting later, not urgent.
- Known caveat, not fixed in this pass: whole-session reliability
  comparisons between saline (short) and DCZ (long) recordings will have a
  trial-count confound baked in, since `combined_reliable`'s thresholds
  aren't duration-adjusted. Interpret saline-vs-DCZ reliability-rate
  differences cautiously until the deferred time-binning work addresses
  this.

Indexing convention (shared with Phases 0/1/2)
-----------------------------------------------
Every per-cell array here (`preproc.h5`'s arrays, Phase 1's `layer_codes`,
this module's own outputs) is indexed by POSITION in
`stat[iscell[:, 0] == 1]` -- i.e. `np.where(iscell[:, 0] == 1)[0]`. No index
translation is ever needed between Phase 1/2/3 outputs for the same
session.
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import glob
from collections import Counter

import numpy as np
import h5py
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # interactive popups (checkbox pickers etc.) need a real GUI backend
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons, Button

rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20

# Existing pipeline code, reused via import -- never modified.
from helper import files
from helper import SMI_Calculation as SMI
from helper.SpatialModulationIndexLayerSpecific import SpatialModulationIndexLayerSpecific as SMI_Layer
from helper.SMICalculation_LayerSpecific_SingleRecording import filter_onset_response_cells
from helper.ReliabilityTesting import (
    evaluate_pattern_similarity_improved,
    improved_activity_threshold_check,
    combined_reliability_test_improved,
)

MICRONS_PER_PIXEL = 1.08952017715202  # V1_prism_DREADD animals (JSY090, JSY093)


# =============================================================================
# Function 3.1 -- load Phase 1's curve-based layer assignment for one session
# =============================================================================

def find_layer_curve_path(plane0_path, prefer_averaged=True):
    """
    Locate a session's Phase 1 layer-curve-results file, given its
    suite2p/plane0 path. Same logic as 2.CellTracking.py's function of the
    same name, reimplemented here since neither Phase 1 nor Phase 2's
    module filenames can be imported via a normal Python import statement
    (digit-prefixed module names).
    """
    tseries_dir = os.path.dirname(os.path.dirname(str(plane0_path)))
    averaged_matches = glob.glob(os.path.join(tseries_dir, '*_layer_curve_results_averaged.h5'))
    independent_matches = glob.glob(os.path.join(tseries_dir, '*_layer_curve_results.h5'))

    if prefer_averaged and averaged_matches:
        matches = averaged_matches
    elif independent_matches:
        matches = independent_matches
    elif averaged_matches:
        matches = averaged_matches
    else:
        raise FileNotFoundError(f"No *_layer_curve_results(_averaged).h5 found in {tseries_dir} "
                                 "-- has Phase 1 been run for this session?")

    if len(matches) > 1:
        print(f"WARNING: multiple matches in {tseries_dir}, using {matches[0]}")
    return matches[0]


def load_session_layer_cells_for_smi(plane0_path, layer_names=('L2/3', 'L4', 'L5', 'L6')):
    """
    Load Phase 1's curve-based layer assignment for one session, converted
    to {layer_name: indices} -- the shape identify_layers() used to produce.

    Parameters
    ----------
    plane0_path : str
        Path to the session's suite2p/plane0 folder.
    layer_names : tuple of str
        Expected layer name order (must match what Phase 1 saved).

    Returns
    -------
    layer_cells : dict
        {layer_name: numpy.ndarray of cell indices}, indices are positions
        in the iscell-filtered cell list -- same convention as
        preproc.h5's arrays.
    """
    layer_curve_path = find_layer_curve_path(plane0_path)

    with h5py.File(layer_curve_path, 'r') as f:
        layer_codes = f['layer_codes'][:]
        saved_layer_names = tuple(n.decode() if isinstance(n, bytes) else n
                                   for n in f['layer_names'][:])

    if saved_layer_names != layer_names:
        print(f"NOTE: saved layer_names {saved_layer_names} differ in order from "
              f"the requested {layer_names} -- using the saved order.")

    layer_cells = {
        name: np.where(layer_codes == code)[0]
        for code, name in enumerate(saved_layer_names)
    }

    print(f"Loaded layer assignment from {layer_curve_path}")
    for name, idx in layer_cells.items():
        print(f"  {name}: {len(idx)} cells")

    return layer_cells


# =============================================================================
# Function 3.2 -- per-session SMI driver (Track B directly: all
# combined_reliable cells in this one session, computed independently)
# =============================================================================

def save_smi_results_dreadd(session_dir, session_label, result):
    """
    Save one session's DREADD SMI result to
    '{TSeries folder name}_smi_results_dreadd.h5' -- a distinct filename so
    it never collides with any pre-existing smi_results.h5 for this
    session. Not a call to the existing save_smi_results: that function's
    layer-boundary section assumes each layer has a flat (upper, lower)
    tuple, which doesn't fit Phase 1's curve-based boundaries (a function
    of x, not a constant). Everything else about it mirrors the existing
    file's group/dataset naming for consistency, and additionally persists
    combined_reliable/avg_cc/cohen_d (so the reliability diagnostic,
    Function 3.3, doesn't need to reload preproc.h5 separately) plus which
    Phase 1 layer-curve file was used, for provenance.

    The filename is built from the TSeries folder's own name, not
    session_label -- session_label can be a long, fully-disambiguated
    catalog key (see discover_animal_sessions, whose collision handling
    appends the entire raw TSeries folder name onto an already-long
    parent-folder-derived base label). Repeating that inside the filename
    on top of an already-long parent path risks exceeding Windows'
    260-character MAX_PATH limit -- hit in practice for one of the
    StationaryOpenLoop sessions. session_dir already scopes the file to
    this exact session, so the TSeries folder's own (much shorter) name is
    just as unique here. session_label is still stored as an attribute
    below, for provenance/display.

    Parameters
    ----------
    session_dir : str
    session_label : str
    result : dict
        The dict run_smi_analysis_session builds internally (before this
        save step is called).

    Returns
    -------
    save_path : str
    """
    smi_results = result['results_full']['smi_results']
    layer_results = result['layer_results']
    layer_cells = result['layer_cells']

    tseries_name = os.path.basename(session_dir)
    save_path = os.path.join(session_dir, f'{tseries_name}_smi_results_dreadd.h5')

    with h5py.File(save_path, 'w') as f:
        f.attrs['session_label'] = session_label
        f.attrs['layer_curve_source'] = result.get('layer_curve_path', '')

        global_grp = f.create_group('global_smi')
        global_grp.create_dataset('SMI_all_cells', data=smi_results['SMI'])
        global_grp.create_dataset('valid_cells_mask', data=smi_results['reliable_valid_cells'])
        global_grp.create_dataset('analysis_reliable_cells', data=result['analysis_reliable_cells'])
        global_grp.create_dataset('preferred_positions', data=smi_results['preferred_positions'])
        global_grp.create_dataset('non_preferred_positions', data=smi_results['non_preferred_positions'])
        global_grp.create_dataset('Rp', data=smi_results['Rp'])
        global_grp.create_dataset('Rn', data=smi_results['Rn'])

        # Reliability info, persisted here so Function 3.3 doesn't need to
        # reload preproc.h5 separately.
        rel_grp = f.create_group('reliability')
        rel_grp.create_dataset('combined_reliable', data=result['combined_reliable'])
        rel_grp.create_dataset('avg_cc', data=result['avg_cc'])
        rel_grp.create_dataset('cohen_d', data=result['cohen_d'])

        coords_grp = f.create_group('cell_info')
        coords_grp.create_dataset('med_coords', data=result['med_coords'])
        coords_grp.create_dataset('bin_centers', data=result['bin_centers'])

        layers_grp = f.create_group('layer_smi')
        for layer_name, cell_indices in layer_cells.items():
            safe_name = layer_name.replace('/', '_')
            layer_grp = layers_grp.create_group(safe_name)
            layer_grp.attrs['original_name'] = layer_name
            layer_grp.create_dataset('cell_indices', data=cell_indices)

            lr = layer_results.get(layer_name)
            if lr is not None:
                layer_grp.create_dataset('reliable_valid_cells', data=lr['reliable_valid_cells'])
                layer_grp.create_dataset('SMI', data=lr['SMI'])
                layer_grp.attrs['n_cells_total'] = len(cell_indices)
                layer_grp.attrs['n_cells_valid'] = len(lr['SMI'])
                layer_grp.attrs['median_smi'] = float(lr['stats']['median'])
                layer_grp.attrs['mean_smi'] = float(lr['stats']['mean'])
                layer_grp.attrs['std_smi'] = float(lr['stats']['std'])
                layer_grp.attrs['sem_smi'] = float(lr['stats']['sem'])
                if lr['preferred_positions'] is not None:
                    layer_grp.create_dataset('preferred_positions', data=lr['preferred_positions'])
                if lr['Rp'] is not None:
                    layer_grp.create_dataset('Rp', data=lr['Rp'])
                if lr['Rn'] is not None:
                    layer_grp.create_dataset('Rn', data=lr['Rn'])
            else:
                layer_grp.attrs['n_cells_total'] = len(cell_indices)
                layer_grp.attrs['n_cells_valid'] = 0

    print(f"Saved DREADD SMI results -> {save_path}")
    return save_path


def run_smi_analysis_session(session_dir, session_label=None,
                              exclude_first_bins=10, exclude_last_bins=10,
                              segment_distance=28, exclude_start_cm=15, exclude_end_cm=10,
                              smoothing_sigma=1.0, save_output=True):
    """
    Per-session SMI driver for one DREADD session (Track B: every
    combined_reliable & non-onset cell in this session, computed
    independently of any other session). Mirrors Run_SMI_Layer_Analysis's
    flow (load preproc.h5 -> onset filtering -> analyze_spatial_modulation_improved
    -> layer bucketing -> save), with layer identification replaced by
    Function 3.1's output. The bin-center rescaling
    (shifted_centers * n_bins/max(shifted_centers)) is kept byte-for-byte
    identical to the original, since segment_distance/exclude_start_cm/
    exclude_end_cm are all calibrated in that rescaled unit space, not raw cm.

    Parameters
    ----------
    session_dir : str
        Path to the TSeries folder (contains suite2p/plane0/ and *preproc*.h5).
    session_label : str, optional
    exclude_first_bins, exclude_last_bins : int
    segment_distance, exclude_start_cm, exclude_end_cm, smoothing_sigma : float
        Same defaults as Run_SMI_Layer_Analysis.
    save_output : bool

    Returns
    -------
    result : dict
        session_label, save_path, results_full, layer_results, layer_cells,
        layer_curve_path, analysis_reliable_cells, combined_reliable,
        avg_cc, cohen_d, med_coords, bin_centers.
    """
    if session_label is None:
        session_label = os.path.basename(session_dir)

    plane0_path = os.path.join(session_dir, 'suite2p', 'plane0')

    # --- Step 1: load preproc.h5 ---
    preproc_files = glob.glob(os.path.join(session_dir, "*preproc*.h5"))
    if not preproc_files:
        raise FileNotFoundError(f"No *preproc*.h5 found in {session_dir}")
    preproc_data = files.read_h5(preproc_files[0])

    spatial_activity = preproc_data['spatial_activity']
    normalized_spatial_activity = preproc_data['norm_spatial_activity']
    bin_centers = preproc_data['bin_centers']
    combined_reliable = preproc_data['combined_reliable']
    avg_cc = preproc_data['avg_cc']
    cohen_d = preproc_data['cohen_d']
    med_coords = preproc_data['med_coords']

    n_cells, n_trials, n_bins = spatial_activity.shape
    print(f"\n{session_label}: {n_cells} cells, {n_trials} trials, {n_bins} bins")
    print(f"  combined_reliable: {np.sum(combined_reliable)} cells")

    # Same bin-center rescaling Run_SMI_Layer_Analysis uses.
    shifted_centers = bin_centers - np.min(bin_centers)
    scaled_bin_centers = shifted_centers * (np.size(bin_centers) / np.max(shifted_centers))

    # --- Step 2: layer assignment (Phase 1's curve-based output) ---
    layer_curve_path = find_layer_curve_path(plane0_path)
    layer_cells = load_session_layer_cells_for_smi(plane0_path)

    # --- Step 3: onset/reward filtering (existing, imported) ---
    non_onset_cells, rejected_info = filter_onset_response_cells(
        spatial_activity, scaled_bin_centers, combined_reliable,
        exclude_first_bins=exclude_first_bins, exclude_last_bins=exclude_last_bins
    )
    analysis_reliable_cells = combined_reliable & non_onset_cells
    print(f"  Final cells for analysis (combined_reliable & non-onset): "
          f"{np.sum(analysis_reliable_cells)}")

    # --- Step 4: SMI calculation (existing, imported, untouched) ---
    results_full = SMI.analyze_spatial_modulation_improved(
        spatial_activity, scaled_bin_centers, analysis_reliable_cells,
        avg_cc=avg_cc, cohens_d=cohen_d,
        segment_distance=segment_distance, exclude_start_cm=exclude_start_cm,
        exclude_end_cm=exclude_end_cm, smoothing_sigma=smoothing_sigma,
        data_filepath=session_dir,
    )

    # --- Step 5: layer-specific bucketing (existing, imported, untouched) ---
    # run_layer_SMI_analysis's internal plot_layer_comparison treats save_path
    # as an existing directory (os.path.join(save_path, "...png")) rather than
    # creating it -- Run_SMI_Layer_Analysis gets away with this because it
    # already created its 'SMI_Figures' dir earlier for onset-filter viz and
    # reuses that. This is a new directory name, so it must be created here.
    viz_dir = os.path.join(session_dir, 'SMI_Figures_DREADD')
    os.makedirs(viz_dir, exist_ok=True)

    valid_cells = results_full['smi_results']['reliable_valid_cells']
    layer_results, _ = SMI_Layer.run_layer_SMI_analysis(
        results_full['smi_results'], valid_cells, med_coords, layer_cells,
        normalized_spatial_activity, scaled_bin_centers,
        save_path=viz_dir
    )

    result = {
        'session_label': session_label,
        'save_path': None,
        'results_full': results_full,
        'layer_results': layer_results,
        'layer_cells': layer_cells,
        'layer_curve_path': layer_curve_path,
        'analysis_reliable_cells': analysis_reliable_cells,
        'combined_reliable': combined_reliable,
        'avg_cc': avg_cc,
        'cohen_d': cohen_d,
        'med_coords': med_coords,
        'bin_centers': scaled_bin_centers,
    }

    if save_output:
        result['save_path'] = save_smi_results_dreadd(session_dir, session_label, result)

    return result


# =============================================================================
# Function 3.3 -- SMI vs. reliability diagnostic (per session)
# =============================================================================

def build_smi_reliability_diagnostic(smi_result, discrepancy_smi_threshold=0.15,
                                      discrepancy_cc_percentile=75):
    """
    Combine per-cell SMI with the already-computed avg_cc/cohen_d/
    combined_reliable (no new computation) into one table, and flag two
    discrepancy patterns:

    - 'reliable_but_low_smi' -- combined_reliable, with avg_cc in the top
      quartile of reliable cells (well above the minimum bar, i.e. a
      genuinely solid, repeatable tuning curve), but SMI below threshold
      anyway. Likely candidates: a landmark-periodicity artifact deflating
      Rn near another bump, or a broad/multi-peaked field that doesn't fit
      SMI's single-peak assumption well.
    - 'high_smi_but_unreliable' -- SMI above threshold despite failing
      combined_reliable. An SMI value that "looks tuned" without passing
      the reliability bar -- worth distrusting rather than taking at face
      value.

    Parameters
    ----------
    smi_result : dict
        From run_smi_analysis_session.
    discrepancy_smi_threshold : float
    discrepancy_cc_percentile : float

    Returns
    -------
    df : pandas.DataFrame
        One row per cell: SMI, avg_cc, cohen_d, combined_reliable, valid,
        layer, discrepancy.
    summary : dict
        Counts.
    """
    smi_results = smi_result['results_full']['smi_results']
    SMI_values = smi_results['SMI']
    valid_cells = smi_results['valid_cells']
    combined_reliable = smi_result['combined_reliable']
    avg_cc = smi_result['avg_cc']
    cohen_d = smi_result['cohen_d']
    layer_cells = smi_result['layer_cells']

    n_cells = len(SMI_values)
    layer_of_cell = np.full(n_cells, None, dtype=object)
    for layer_name, idx in layer_cells.items():
        layer_of_cell[idx] = layer_name

    df = pd.DataFrame({
        'cell_idx': np.arange(n_cells),
        'SMI': SMI_values,
        'avg_cc': avg_cc,
        'cohen_d': cohen_d,
        'combined_reliable': combined_reliable,
        'valid': valid_cells,
        'layer': layer_of_cell,
    })

    cc_threshold = (np.percentile(avg_cc[combined_reliable], discrepancy_cc_percentile)
                    if combined_reliable.any() else np.inf)

    reliable_low_smi = (df['combined_reliable'] & (df['avg_cc'] >= cc_threshold) &
                         (df['SMI'] < discrepancy_smi_threshold) & df['valid'])
    high_smi_unreliable = ((~df['combined_reliable']) & (df['SMI'] >= discrepancy_smi_threshold)
                            & df['valid'])

    df['discrepancy'] = None
    df.loc[reliable_low_smi, 'discrepancy'] = 'reliable_but_low_smi'
    df.loc[high_smi_unreliable, 'discrepancy'] = 'high_smi_but_unreliable'

    summary = {
        'n_cells': n_cells,
        'n_reliable_but_low_smi': int(reliable_low_smi.sum()),
        'n_high_smi_but_unreliable': int(high_smi_unreliable.sum()),
        'cc_threshold_used': float(cc_threshold),
    }

    print(f"Reliability diagnostic for {smi_result['session_label']}:")
    print(f"  reliable_but_low_smi: {summary['n_reliable_but_low_smi']} cells "
          f"(combined_reliable, avg_cc >= {cc_threshold:.3f}, SMI < {discrepancy_smi_threshold})")
    print(f"  high_smi_but_unreliable: {summary['n_high_smi_but_unreliable']} cells "
          f"(SMI >= {discrepancy_smi_threshold}, not combined_reliable)")

    return df, summary


def plot_smi_reliability_scatter(df, session_label='', save_path=None):
    """
    SMI vs. avg_cc scatter, colored by combined_reliable, discrepancy cells
    outlined -- the visual complement to build_smi_reliability_diagnostic.

    Parameters
    ----------
    df : pandas.DataFrame
        From build_smi_reliability_diagnostic.
    session_label : str
    save_path : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    plot_df = df[df['valid']]

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = np.where(plot_df['combined_reliable'], '#2ca02c', '#7f7f7f')
    ax.scatter(plot_df['avg_cc'], plot_df['SMI'], c=colors, s=25, alpha=0.6, label=None)

    for discrepancy_type, marker, edgecolor in [
        ('reliable_but_low_smi', 'D', 'blue'),
        ('high_smi_but_unreliable', '^', 'red'),
    ]:
        flagged = plot_df[plot_df['discrepancy'] == discrepancy_type]
        if len(flagged) > 0:
            ax.scatter(flagged['avg_cc'], flagged['SMI'], facecolors='none',
                       edgecolors=edgecolor, s=90, linewidths=1.5, marker=marker,
                       label=f'{discrepancy_type} (n={len(flagged)})')

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('avg_cc (split-half correlation)')
    ax.set_ylabel('SMI')
    ax.set_title(f'{session_label}: SMI vs. reliability')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=10, label='combined_reliable'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#7f7f7f', markersize=10, label='not reliable'),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=legend_elements + handles, loc='upper left', fontsize=13)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reliability scatter -> {save_path}")

    return fig


# =============================================================================
# Function 3.3b -- why did each non-combined_reliable cell fail?
# =============================================================================

def diagnose_reliability_failure_reasons(session_dir, smi_result,
                                          min_cc_threshold=0.1, cohen_threshold=0.8,
                                          min_pattern_corr=0.3, peak_distance_threshold=5,
                                          activity_method='absolute_percentile'):
    """
    Break down why each non-combined_reliable cell failed.
    combined_reliable = reliable_cells & pattern_reliable & active_cells,
    and reliable_cells itself requires avg_cc > min_cc_threshold AND avg_cc
    beats its own shuffle-null AND cohen_d > cohen_threshold.

    - 'below_cc_floor', 'below_cohen_d' -- simple threshold checks on the
      already-saved avg_cc/cohen_d, no recomputation needed.
    - 'inactive', 'pattern_unstable' -- both fully deterministic (no
      shuffling), recomputed directly from preproc.h5's spatial_activity
      via improved_activity_threshold_check/evaluate_pattern_similarity_improved
      (imported, existing code).
    - 'below_shuffle_significance_inferred' -- the one part that genuinely
      isn't recomputable without rerunning n_shuffles=300 shuffles
      (per-cell significance vs. its own null). Inferred by elimination: a
      cell that passes both flat floors, is active, and has a stable
      pattern, but is still not reliable_cells, must have failed
      specifically the shuffle-significance check.

    CAVEAT: the threshold defaults here (min_cc_threshold=0.1,
    cohen_threshold=0.8, min_pattern_corr=0.3, peak_distance_threshold=5)
    must match whatever Preprocess.py actually used for this session's
    area -- these are the area='V1' defaults; double-check if unsure.

    Parameters
    ----------
    session_dir : str
    smi_result : dict
        From run_smi_analysis_session.
    min_cc_threshold, cohen_threshold, min_pattern_corr, peak_distance_threshold : float
        Must match Preprocess.py's actual call for this session's area.
    activity_method : str

    Returns
    -------
    df : pandas.DataFrame
        One row per cell (all cells, flagged with combined_reliable so you
        can filter): avg_cc, cohen_d, active, pattern_reliable,
        odd_even_corr, peak_distance, and boolean failure-category columns.
    summary : dict
        Counts per failure category, among cells failing combined_reliable.
    """
    preproc_files = glob.glob(os.path.join(session_dir, "*preproc*.h5"))
    if not preproc_files:
        raise FileNotFoundError(f"No *preproc*.h5 found in {session_dir}")
    preproc_data = files.read_h5(preproc_files[0])
    spatial_activity = preproc_data['spatial_activity']

    combined_reliable = smi_result['combined_reliable']
    avg_cc = smi_result['avg_cc']
    cohen_d = smi_result['cohen_d']

    active_cells, _ = improved_activity_threshold_check(spatial_activity, method=activity_method)
    pattern_reliable, odd_even_corr, peak_distances = evaluate_pattern_similarity_improved(
        spatial_activity, min_pattern_corr, peak_distance_threshold
    )

    n_cells = len(avg_cc)
    below_cc_floor = avg_cc < min_cc_threshold
    below_cohen_d = cohen_d < cohen_threshold
    inactive = ~active_cells
    pattern_unstable = ~pattern_reliable

    # Cells failing reliable_cells despite passing both flat floors, being
    # active, and having a stable pattern -- the only remaining explanation
    # is the shuffle-significance check specifically (not recomputed here).
    below_shuffle_significance = (
        (~below_cc_floor) & (~below_cohen_d) & active_cells & pattern_reliable & (~combined_reliable)
    )

    df = pd.DataFrame({
        'cell_idx': np.arange(n_cells),
        'combined_reliable': combined_reliable,
        'avg_cc': avg_cc,
        'cohen_d': cohen_d,
        'active': active_cells,
        'pattern_reliable': pattern_reliable,
        'odd_even_corr': odd_even_corr,
        'peak_distance': peak_distances,
        'below_cc_floor': below_cc_floor,
        'below_cohen_d': below_cohen_d,
        'inactive': inactive,
        'pattern_unstable': pattern_unstable,
        'below_shuffle_significance_inferred': below_shuffle_significance,
    })

    failed_df = df[~df['combined_reliable']]

    summary = {
        'n_failed': len(failed_df),
        'below_cc_floor': int(failed_df['below_cc_floor'].sum()),
        'below_cohen_d': int(failed_df['below_cohen_d'].sum()),
        'inactive': int(failed_df['inactive'].sum()),
        'pattern_unstable': int(failed_df['pattern_unstable'].sum()),
        'below_shuffle_significance_inferred': int(failed_df['below_shuffle_significance_inferred'].sum()),
    }

    print(f"Of {summary['n_failed']} cells failing combined_reliable "
          f"(a cell can appear in multiple categories):")
    print(f"  below_cc_floor (avg_cc < {min_cc_threshold}): {summary['below_cc_floor']}")
    print(f"  below_cohen_d (cohen_d < {cohen_threshold}): {summary['below_cohen_d']}")
    print(f"  inactive (fails activity threshold): {summary['inactive']}")
    print(f"  pattern_unstable (odd/even peak/corr, 'fixed' test): {summary['pattern_unstable']}")
    print(f"  below_shuffle_significance (inferred by elimination): "
          f"{summary['below_shuffle_significance_inferred']}")

    return df, summary


# =============================================================================
# Phase 0 sensitivity check (kept for reference, not part of the main flow)
# -- pattern_test='shuffled' vs. the default 'fixed', re-run on
# already-saved spatial_activity. Result on the Day1 test session: WORSE
# (299 -> 235 combined_reliable) -- combined_reliable was kept as-is.
# Retained here in case this needs revisiting for a different session/area,
# not because it's expected to be run routinely.
# =============================================================================

def run_reliability_sensitivity_check(session_dir, pattern_test='shuffled',
                                       n_shuffles=300, cc_percentile=90,
                                       cohen_threshold=0.8, min_cc_threshold=0.1,
                                       min_pattern_corr=0.3, peak_distance_threshold=5,
                                       pattern_percentile=95, peak_distance_percentile=95,
                                       use_activity_threshold=True,
                                       activity_method='absolute_percentile'):
    """
    Rerun just the reliability test (not the whole Preprocess.py pipeline)
    on this session's already-saved spatial_activity, with a chosen
    pattern_test setting.

    Parameters
    ----------
    session_dir : str
    pattern_test : str
        'shuffled' (the alternative tested) or 'fixed' (the current
        default, for a same-call comparison baseline).
    Remaining parameters : same values Preprocess.py used for area='V1',
        so only pattern_test differs from what's already in preproc.h5.

    Returns
    -------
    result : dict
        combined_reliable, reliable_cells, pattern_reliable, avg_cc,
        cohen_d, odd_even_corr, peak_distances, active_cells.
    """
    preproc_files = glob.glob(os.path.join(session_dir, "*preproc*.h5"))
    if not preproc_files:
        raise FileNotFoundError(f"No *preproc*.h5 found in {session_dir}")
    preproc_data = files.read_h5(preproc_files[0])
    spatial_activity = preproc_data['spatial_activity']

    (combined_reliable, reliable_cells, pattern_reliable, avg_cc, cohen_d,
     odd_even_corr, peak_distances, active_cells) = combined_reliability_test_improved(
        spatial_activity,
        n_shuffles=n_shuffles,
        cc_percentile=cc_percentile,
        cohen_threshold=cohen_threshold,
        min_cc_threshold=min_cc_threshold,
        min_pattern_corr=min_pattern_corr,
        peak_distance_threshold=peak_distance_threshold,
        use_activity_threshold=use_activity_threshold,
        activity_method=activity_method,
        pattern_test=pattern_test,
        pattern_percentile=pattern_percentile,
        peak_distance_percentile=peak_distance_percentile,
    )

    return {
        'combined_reliable': combined_reliable,
        'reliable_cells': reliable_cells,
        'pattern_reliable': pattern_reliable,
        'avg_cc': avg_cc,
        'cohen_d': cohen_d,
        'odd_even_corr': odd_even_corr,
        'peak_distances': peak_distances,
        'active_cells': active_cells,
    }


def plot_newly_recovered_cells(session_dir, newly_recovered_mask, n_examples=8, seed=0):
    """
    Plot odd-vs-even mean tuning curves for a random sample of newly-
    recovered cells, so you can eyeball whether they look like genuine
    place fields or noise before trusting the sensitivity check's counts
    alone.

    Parameters
    ----------
    session_dir : str
    newly_recovered_mask : numpy.ndarray of bool
    n_examples : int
    seed : int

    Returns
    -------
    fig : matplotlib.figure.Figure or None (if nothing to plot)
    """
    preproc_files = glob.glob(os.path.join(session_dir, "*preproc*.h5"))
    preproc_data = files.read_h5(preproc_files[0])
    spatial_activity = preproc_data['spatial_activity']
    bin_centers = preproc_data['bin_centers']

    idx = np.where(newly_recovered_mask)[0]
    if len(idx) == 0:
        print("No newly recovered cells to plot.")
        return None

    rng = np.random.default_rng(seed)
    sample = rng.choice(idx, size=min(n_examples, len(idx)), replace=False)

    n_trials = spatial_activity.shape[1]
    odd_trials = np.arange(0, n_trials, 2)
    even_trials = np.arange(1, n_trials, 2)

    n_cols = 4
    n_rows = int(np.ceil(len(sample) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, cell in zip(axes, sample):
        odd_mean = np.mean(spatial_activity[cell, odd_trials], axis=0)
        even_mean = np.mean(spatial_activity[cell, even_trials], axis=0)
        ax.plot(bin_centers, odd_mean, label='odd trials', color='steelblue')
        ax.plot(bin_centers, even_mean, label='even trials', color='coral')
        ax.set_title(f"Cell {cell}")
        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Activity')
        ax.legend(fontsize=10)

    for ax in axes[len(sample):]:
        ax.axis('off')

    fig.suptitle(f"{len(sample)} randomly-sampled newly-recovered cells (odd vs even trials)",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# Function 3.4 -- Track A: join a Phase 2 tracking group onto SMI results
#
# Active as of JSY093's DCZ1/DCZ2/DCZ3 tracking groups -- built and
# validated (see notebook history), now wired up via Function 3.6
# (load_track_a_group) below. Entry point: get master_df from
# 2.CellTracking.py's track_and_build_table/build_master_cell_table (here,
# reloaded from its saved '{group_name}_master_table.csv'), get
# smi_results_by_label by reloading each session's already-saved
# *_smi_results_dreadd.h5 (Function 3.2/run_smi_batch_track_b already
# computed these -- Track A recomputes no SMI, it only joins), then call
# this. See load_track_a_group for the assembled version of this pattern.
# =============================================================================

def join_smi_to_master_table(master_df, smi_results_by_label):
    """
    Join a Phase 2 tracking group's master_df to each session's SMI output,
    using the roi_idx_<label> columns master_df already has (Phase 2's
    build_master_cell_table) as the join key -- same iscell-filtered-
    position convention used throughout this whole pipeline, no
    translation needed.

    Takes master_df as a plain in-memory DataFrame rather than reloading
    anything from Phase 2's saved .h5 itself, keeping this function fully
    decoupled from Phase 2's internals.

    Parameters
    ----------
    master_df : pandas.DataFrame
        From Phase 2's build_master_cell_table (optionally already passed
        through flag_layer_mismatches).
    smi_results_by_label : dict
        {label: run_smi_analysis_session(...) result}, one entry per
        session in the group.

    Returns
    -------
    df : pandas.DataFrame
        master_df plus new SMI_<label>, valid_<label>,
        combined_reliable_<label>, analysis_reliable_<label>,
        avg_cc_<label> columns per session. valid_<label> (reliable_valid_
        cells: reliable AND the SMI curve fit succeeded) is the same filter
        Phase 4's Function 4.7 uses for Track B's actual hypothesis test --
        join it (not just analysis_reliable, a looser pre-curve-fit filter)
        so a later Track A paired comparison can filter on the identical
        criterion Track B's comparison already uses.
    """
    df = master_df.copy()

    roi_idx_cols = [c for c in df.columns if c.startswith('roi_idx_')]
    labels = [c[len('roi_idx_'):] for c in roi_idx_cols]

    for label in labels:
        if label not in smi_results_by_label:
            print(f"WARNING: no SMI result provided for '{label}' -- skipping.")
            continue

        smi_result = smi_results_by_label[label]
        SMI_values = smi_result['results_full']['smi_results']['SMI']
        valid_cells = smi_result['results_full']['smi_results']['valid_cells']
        combined_reliable = smi_result['combined_reliable']
        avg_cc = smi_result['avg_cc']
        analysis_reliable_cells = smi_result['analysis_reliable_cells']

        roi_idx = df[f'roi_idx_{label}'].to_numpy()

        df[f'SMI_{label}'] = SMI_values[roi_idx]
        df[f'valid_{label}'] = valid_cells[roi_idx]
        df[f'combined_reliable_{label}'] = combined_reliable[roi_idx]
        df[f'analysis_reliable_{label}'] = analysis_reliable_cells[roi_idx]
        df[f'avg_cc_{label}'] = avg_cc[roi_idx]

    n_reliable_all = np.ones(len(df), dtype=bool)
    n_joined = 0
    for label in labels:
        if f'analysis_reliable_{label}' in df.columns:
            n_reliable_all &= df[f'analysis_reliable_{label}'].to_numpy()
            n_joined += 1

    print(f"Joined SMI for {len(labels)} sessions onto {len(df)} tracked cells.")
    for label in labels:
        if f'analysis_reliable_{label}' in df.columns:
            print(f"  analysis_reliable in {label}: {df[f'analysis_reliable_{label}'].sum()}/{len(df)}")

    if n_joined < len(labels):
        # A missing SMI result (see the WARNINGs above) silently drops that
        # session from the AND below -- make sure that's not mistaken for a
        # true "reliable in every session in the group" figure.
        print(f"  WARNING: only {n_joined}/{len(labels)} sessions actually had SMI data joined "
              f"-- the figure below reflects just those {n_joined} session(s), NOT all "
              f"{len(labels)} in the group. Re-run once every session's "
              f"*_smi_results_dreadd.h5 exists before trusting this as the paired-comparison N.")
    print(f"  analysis_reliable in all {n_joined}/{len(labels)} joined sessions "
          f"(usable for a paired within-cell comparison): {n_reliable_all.sum()}/{len(df)}")

    return df


# =============================================================================
# Function 3.5 -- Track B session picker + batch runner
#
# This is the active entry point for Phase 3 right now. Pick however many
# sessions you want SMI computed for, and run them all in one call.
# =============================================================================

def discover_animal_sessions(animal_dir):
    """
    Scan animal_dir for every TSeries-*/suite2p/plane0 folder anywhere in
    its tree, labeling each by whichever known naming pattern it matches.

    Nothing is silently dropped on a label collision (e.g. a day with both
    a NO_REWARD probe and a normal recording, or a session split into
    sal-001/sal-002 parts) -- every colliding entry is disambiguated by
    appending its raw TSeries folder name, so all of them show up in the
    picker and you choose which one(s) belong.

    Parameters
    ----------
    animal_dir : str

    Returns
    -------
    catalog : dict
        {label: {'plane0_path': str, 'session_type': str, 'tseries_dir': str}}
    """
    plane0_paths = sorted(glob.glob(os.path.join(animal_dir, '**', 'suite2p', 'plane0'),
                                     recursive=True))

    entries = []  # (base_label, session_type, plane0_path, tseries_dir, tseries_name)
    skipped = []
    unmatched = []

    for plane0_path in plane0_paths:
        tseries_dir = os.path.dirname(os.path.dirname(plane0_path))
        tseries_name = os.path.basename(tseries_dir)
        parent_dir = os.path.dirname(tseries_dir)
        parent_name = os.path.basename(parent_dir)

        if 'skip' in tseries_name.lower() or 'skip' in parent_name.lower():
            skipped.append(tseries_dir)
            continue

        upper_tseries = tseries_name.upper()

        if 'SAL' in upper_tseries:
            session_type = 'saline'
            base_label = f'{parent_name}_SALINE'
        elif 'DCZ' in upper_tseries:
            session_type = 'dcz'
            base_label = f'{parent_name}_DCZ'
        else:
            day_match = re.search(r'Day(\d+)', parent_name, re.IGNORECASE)
            if day_match:
                session_type = 'baseline'
                base_label = f'Day{day_match.group(1)}'
            else:
                session_type = 'unknown'
                base_label = tseries_name
                unmatched.append(base_label)

        entries.append((base_label, session_type, plane0_path, tseries_dir, tseries_name))

    # Disambiguate every base_label that collides -- suffix ALL of its
    # entries with the raw TSeries folder name (not just the extras), so
    # the labeling is consistent whether or not a collision happened.
    label_counts = Counter(e[0] for e in entries)

    catalog = {}
    for base_label, session_type, plane0_path, tseries_dir, tseries_name in entries:
        if label_counts[base_label] > 1:
            label = f'{base_label}__{tseries_name}'
        else:
            label = base_label

        if label in catalog:
            print(f"WARNING: label '{label}' still collides after disambiguation -- "
                  f"keeping {catalog[label]['tseries_dir']}, skipping {tseries_dir}")
            continue

        catalog[label] = {
            'plane0_path': plane0_path,
            'session_type': session_type,
            'tseries_dir': tseries_dir,
        }

    print(f"Discovered {len(catalog)} sessions under {animal_dir}:")
    for label, info in catalog.items():
        print(f"  [{info['session_type']:>8}] {label}  <-  {info['tseries_dir']}")

    collided_labels = [l for l, c in label_counts.items() if c > 1]
    if collided_labels:
        print(f"\n{len(collided_labels)} label(s) had multiple TSeries and were disambiguated "
              f"-- review these and pick the right one(s) in the popup: {collided_labels}")

    if skipped:
        print(f"\n{len(skipped)} session(s) excluded via 'skip' in their folder name:")
        for s in skipped:
            print(f"  {s}")

    if unmatched:
        print(f"\n{len(unmatched)} session(s) didn't match a known naming pattern "
              f"(labeled 'unknown', tagged by raw TSeries folder name): {unmatched}")

    return catalog


def select_sessions_popup(session_catalog,
                           title='Select sessions to run SMI on (Track B -- independent per session)'):
    """
    Checkbox popup listing every session in session_catalog. Same UI as
    Phase 2's function of the same name. Track B needs no reference/
    grouping choice at all -- just pick however many sessions you want SMI
    run on, each fully independently.

    Parameters
    ----------
    session_catalog : dict
        From discover_animal_sessions.
    title : str

    Returns
    -------
    selected_labels : list of str
        Labels you checked, in session_catalog's original order.
    """
    labels = list(session_catalog.keys())
    n = len(labels)
    display_labels = [f"[{session_catalog[l]['session_type']:>8}] {l}" for l in labels]

    fig_height = max(4, 0.35 * n + 1.5)
    fig = plt.figure(figsize=(9, fig_height))
    try:
        fig.canvas.manager.set_window_title('SELECT SESSIONS (Track B) -- check boxes, then Confirm (or Enter)')
    except Exception:
        pass

    fig.suptitle(title, fontsize=12, fontweight='bold')

    check_ax = fig.add_axes([0.05, 0.12, 0.9, 0.80])
    check = CheckButtons(check_ax, display_labels, [False] * n)

    confirm_ax = fig.add_axes([0.35, 0.02, 0.3, 0.06])
    confirm_button = Button(confirm_ax, 'Confirm selection')

    state = {'done': False, 'status': [False] * n}

    def on_confirm(event=None):
        state['status'] = list(check.get_status())
        state['done'] = True
        fig.canvas.stop_event_loop()

    confirm_button.on_clicked(on_confirm)

    def on_key(event):
        if event.key == 'enter':
            on_confirm()

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show(block=False)
    while not state['done'] and plt.fignum_exists(fig.number):
        fig.canvas.start_event_loop(0.1)

    if not state['done']:
        state['status'] = list(check.get_status())
        print("Window closed without pressing Confirm -- using current checkbox state anyway.")

    if plt.fignum_exists(fig.number):
        plt.close(fig)
        plt.pause(0.01)  # let Qt actually process the close/hide before we return

    selected_labels = [label for label, checked in zip(labels, state['status']) if checked]
    print(f"Selected {len(selected_labels)} sessions: {selected_labels}")

    return selected_labels


def run_smi_batch_track_b(session_catalog, selected_labels, **smi_kwargs):
    """
    Loop run_smi_analysis_session over the checked sessions, each fully
    independently (Track B).

    Parameters
    ----------
    session_catalog : dict
        From discover_animal_sessions.
    selected_labels : list of str
        From select_sessions_popup (or any list of labels you already have).
    **smi_kwargs
        Passed through to run_smi_analysis_session (e.g. segment_distance,
        exclude_start_cm, ...).

    Returns
    -------
    results_by_label : dict
        {label: run_smi_analysis_session(...) result}.
    """
    results_by_label = {}

    for label in selected_labels:
        session_dir = os.path.dirname(os.path.dirname(session_catalog[label]['plane0_path']))
        print(f"\n{'='*90}\nTRACK B -- {label}\n{'='*90}")
        results_by_label[label] = run_smi_analysis_session(
            session_dir, session_label=label, **smi_kwargs
        )

    print(f"\n{'='*90}\nBatch complete: {len(results_by_label)}/{len(selected_labels)} sessions")
    for label, result in results_by_label.items():
        n_reliable = np.sum(result['analysis_reliable_cells'])
        n_total = len(result['analysis_reliable_cells'])
        print(f"  {label}: {n_reliable}/{n_total} analysis-reliable cells")

    return results_by_label


# =============================================================================
# Reload helper -- for when you want smi_results_by_label from disk instead
# of keeping run_smi_batch_track_b's return value in memory (e.g. picking
# Function 3.4's Track A join back up in a later session). Reads exactly
# what save_smi_results_dreadd wrote, in the shape join_smi_to_master_table
# and the diagnostics above expect.
# =============================================================================

def load_smi_results_dreadd(save_path):
    """
    Reload one session's *_smi_results_dreadd.h5 back into the same dict
    shape run_smi_analysis_session returns in memory (minus results_full's
    figures, which aren't persisted -- only results_full['smi_results']
    with the fields this module's downstream functions actually read:
    SMI, valid_cells, reliable_valid_cells, preferred_positions,
    non_preferred_positions, Rp, Rn).

    Parameters
    ----------
    save_path : str
        Path to a *_smi_results_dreadd.h5 file (from save_smi_results_dreadd).

    Returns
    -------
    result : dict -- same keys as run_smi_analysis_session's return value,
        with 'layer_results' rebuilt just enough for join_smi_to_master_table
        and the diagnostics to work (SMI/reliable_valid_cells/cell_indices
        per layer; the full stats dict from SMI_Layer.run_layer_SMI_analysis
        is not reconstructed since it isn't persisted).
    """
    with h5py.File(save_path, 'r') as f:
        session_label = f.attrs.get('session_label', os.path.basename(save_path))
        layer_curve_path = f.attrs.get('layer_curve_source', '')

        SMI_values = f['global_smi/SMI_all_cells'][:]
        valid_cells_mask = f['global_smi/valid_cells_mask'][:]
        analysis_reliable_cells = f['global_smi/analysis_reliable_cells'][:]
        preferred_positions = f['global_smi/preferred_positions'][:]
        non_preferred_positions = f['global_smi/non_preferred_positions'][:]
        Rp = f['global_smi/Rp'][:]
        Rn = f['global_smi/Rn'][:]

        combined_reliable = f['reliability/combined_reliable'][:]
        avg_cc = f['reliability/avg_cc'][:]
        cohen_d = f['reliability/cohen_d'][:]

        med_coords = f['cell_info/med_coords'][:]
        bin_centers = f['cell_info/bin_centers'][:]

        layer_cells = {}
        for safe_name in f['layer_smi']:
            layer_grp = f['layer_smi'][safe_name]
            original_name = layer_grp.attrs.get('original_name', safe_name)
            layer_cells[original_name] = layer_grp['cell_indices'][:]

    smi_results = {
        'SMI': SMI_values,
        'valid_cells': valid_cells_mask,
        'reliable_valid_cells': valid_cells_mask,
        'preferred_positions': preferred_positions,
        'non_preferred_positions': non_preferred_positions,
        'Rp': Rp,
        'Rn': Rn,
    }

    return {
        'session_label': session_label,
        'save_path': save_path,
        'results_full': {'smi_results': smi_results},
        'layer_results': None,  # not reconstructed from disk -- rerun run_smi_analysis_session if needed
        'layer_cells': layer_cells,
        'layer_curve_path': layer_curve_path,
        'analysis_reliable_cells': analysis_reliable_cells,
        'combined_reliable': combined_reliable,
        'avg_cc': avg_cc,
        'cohen_d': cohen_d,
        'med_coords': med_coords,
        'bin_centers': bin_centers,
    }


# =============================================================================
# Function 3.6 -- Track A driver: assemble + join a tracking group's SMI
# table, and save it. Reimplements the tiny master_df reload directly (not
# imported from 2.CellTracking.py -- digit-prefixed module names aren't
# cleanly importable, same convention as Function 3.1's
# find_layer_curve_path above) rather than pulling in Phase 2's module.
# =============================================================================

def load_track_a_master_table(tracked_groups_dir, group_name):
    """
    Reload one Phase 2 tracking group's master_df, saved by
    2.CellTracking.py's save_master_cell_table as
    '{group_name}_master_table.csv'. Just a pd.read_csv -- reimplemented
    here rather than imported (see module-level note above).

    Parameters
    ----------
    tracked_groups_dir : str
        The TrackedGroups folder Phase 2 saved this group into (next to
        the reference session's TSeries directory, unless a custom
        store_dir was passed to track_and_build_table).
    group_name : str

    Returns
    -------
    master_df : pandas.DataFrame
    """
    table_path = os.path.join(tracked_groups_dir, f'{group_name}_master_table.csv')
    master_df = pd.read_csv(table_path)
    print(f"Loaded master table <- {table_path} ({len(master_df)} rows)")
    return master_df


def load_track_a_group(session_catalog, group_name, tracked_groups_dir):
    """
    Track A entry point: reload one Phase 2 tracking group's master_df and
    every session's already-computed Track B SMI result, and join them.

    Recomputes no SMI -- SMI is a per-session, per-cell quantity Track B
    already computed and saved for every session in this cohort. This just
    looks up each tracked cell's already-existing SMI value in each
    session via that session's roi_idx_<label> (the position Phase 2's
    tracking assigned it to in that session), producing one row per
    tracked cell with SMI_<label> for every session in the group side by
    side -- see join_smi_to_master_table.

    Parameters
    ----------
    session_catalog : dict
        From this module's discover_animal_sessions, run on the same
        animal_dir the tracking group came from -- used to locate each
        session's saved *_smi_results_dreadd.h5.
    group_name : str
        e.g. 'DCZ1' -- must match the group_name track_and_build_table was
        called with in 2.CellTracking.py.
    tracked_groups_dir : str
        Same TrackedGroups folder load_track_a_master_table reads from.

    Returns
    -------
    joined_df : pandas.DataFrame
        master_df plus SMI_<label>/combined_reliable_<label>/
        analysis_reliable_<label>/avg_cc_<label> per session.
    """
    master_df = load_track_a_master_table(tracked_groups_dir, group_name)

    roi_idx_cols = [c for c in master_df.columns if c.startswith('roi_idx_')]
    labels = [c[len('roi_idx_'):] for c in roi_idx_cols]

    smi_results_by_label = {}
    for label in labels:
        if label not in session_catalog:
            print(f"WARNING: '{label}' not found in session_catalog -- can't locate its "
                  f"*_smi_results_dreadd.h5, skipping.")
            continue

        # Glob rather than construct the exact filename from the CURRENT
        # folder name -- save_smi_results_dreadd names the file from
        # os.path.basename(session_dir) at the time Track B was run, which
        # can go stale if the TSeries folder gets renamed afterward (hit in
        # practice: DCZ2's saline TSeries folder was renamed to add a
        # '_SAL' suffix after Track B already saved
        # 'TSeries-07262026-0755-001_smi_results_dreadd.h5' inside it --
        # same convention find_layer_curve_path already uses for this exact
        # reason).
        tseries_dir = session_catalog[label]['tseries_dir']
        matches = glob.glob(os.path.join(tseries_dir, '*_smi_results_dreadd.h5'))

        if not matches:
            print(f"WARNING: no *_smi_results_dreadd.h5 found in {tseries_dir} for '{label}' "
                  f"-- has Track B's run_smi_analysis_session been run for this session yet? "
                  f"Skipping.")
            continue
        if len(matches) > 1:
            print(f"WARNING: multiple *_smi_results_dreadd.h5 in {tseries_dir}, using {matches[0]}")

        smi_results_by_label[label] = load_smi_results_dreadd(matches[0])

    joined_df = join_smi_to_master_table(master_df, smi_results_by_label)
    return joined_df


def save_track_a_joined_table(joined_df, group_name, save_dir):
    """
    Save one Track A group's joined per-cell SMI table to
    '{group_name}_track_a_smi.csv' in save_dir -- same TrackedGroups
    folder Phase 2 saved this group's master table into.

    Parameters
    ----------
    joined_df : pandas.DataFrame
        From load_track_a_group.
    group_name : str
    save_dir : str

    Returns
    -------
    save_path : str
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{group_name}_track_a_smi.csv')
    joined_df.to_csv(save_path, index=False)
    print(f"Saved Track A joined SMI table -> {save_path}")
    return save_path


# =============================================================================
# Driver
# =============================================================================

if __name__ == '__main__':
    ANIMAL_DIR = r"D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD"

    # ---- Shared setup (used by both tracks) ----
    session_catalog = discover_animal_sessions(ANIMAL_DIR)

    # =========================================================================
    # TRACK B -- population-level, independent per session
    # =========================================================================
    # Already run for every DCZ1/2/3 session (saline + dcz, all 6) -- skipped
    # here so this run goes straight to TRACK A below without blocking on the
    # interactive popup. Uncomment to (re)run Track B for additional/new
    # sessions:
    selected_labels = select_sessions_popup(session_catalog)
    track_b_results = run_smi_batch_track_b(session_catalog, selected_labels,exclude_first_bins=10, exclude_last_bins=10)

    # Optional per-session reliability diagnostic (methods check, not required
    # for every run -- see the "Decisions locked in" note at the top of this
    # file for what was already concluded on the Day1 test session):
    
    # for label, result in track_b_results.items():
    #     diag_df, diag_summary = build_smi_reliability_diagnostic(result)
    #     plot_smi_reliability_scatter(diag_df, session_label=label)
        # plt.show()

    # =========================================================================
    # TRACK A -- tracked-cell, cross-session (joins Track B's already-computed
    # SMI onto each tracking group's master_df -- no new SMI computation, see
    # load_track_a_group)
    # =========================================================================
    # reference_label per group must match 2.CellTracking.py's
    # TRACKING_GROUPS exactly, since that's whose TSeries dir holds the
    # TrackedGroups/ folder each group's master table + tracking .h5 were
    # saved into (save_tracking_group's default save_dir).
    # TRACK_A_GROUPS = [
    #     {
    #         'group_name': 'DCZ1',
    #         'reference_label': '260724_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_1_SALINE',
    #     },
    #     {
    #         'group_name': 'DCZ2',
    #         'reference_label': '260726_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_2_SALINE',
    #     },
    #     {
    #         'group_name': 'DCZ3',
    #         'reference_label': '260728_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_3_SALINE',
    #     },
    # ]

    # track_a_results = {}
    # for cfg in TRACK_A_GROUPS:
    #     tracked_groups_dir = os.path.join(
    #         session_catalog[cfg['reference_label']]['tseries_dir'], 'TrackedGroups'
    #     )
    #     joined_df = load_track_a_group(session_catalog, cfg['group_name'], tracked_groups_dir)
    #     save_track_a_joined_table(joined_df, cfg['group_name'], tracked_groups_dir)
    #     track_a_results[cfg['group_name']] = joined_df
