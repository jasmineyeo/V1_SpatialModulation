"""
Phase 6 -- Landmark Preference Subpopulation (DREADD saline/DCZ cohort)
==========================================================================

Consolidated from 6.LandmarkPreference.ipynb once every function below
worked end-to-end on real JSY090 and JSY093 data.

Scope
-----
Track B only, same reasoning as Phases 4/5 -- Track A's version is
deferred until Track A gets revisited across Phases 3-6 after this phase.

RSC is classically implicated in landmark/boundary-vector spatial coding.
This phase set out to ask whether DCZ's effect on V1 population SMI
(established in Phase 4, decomposed by anatomical layer in Phase 5) is
concentrated in the subpopulation of V1 cells whose spatial firing is
landmark-anchored. It ended up somewhere more specific: a well-
characterized, statistically robust, but causally unresolved and
non-replicating pattern in landmark-4 (reward-proximal) preference
specifically -- see Findings below.

Reuses identify_landmark_responses and plot_cells_by_landmark_assignment
from 3.LandmarkPreference/LandmarkPrefernce_SingleSessionAnalysis.py
UNCHANGED, via import -- that file's name doesn't start with a digit, so
it's importable, unlike this project's own phase-numbered files.

Landmark config for this cohort's VR track: landmark_positions=[37, 65,
93, 120] cm, windows {before:25, after:0} for all four -- corrected from
an initial [25,55,85,115]/{before:15-20,after:10} config after Function
6.0's diagnostic plot showed the original landmark-1 window was picking
up onset-adjacent activity rather than a genuine landmark response
(matching the same swap made directly in
3.LandmarkPreference/LandmarkPrefernce_SingleSessionAnalysis.py's own
__main__ block).

IMPORTANT: landmark identification needs the RAW (cm-scale) bin_centers
straight from preproc.h5 -- NOT Phase 3's saved *_smi_results_dreadd.h5,
whose stored bin_centers is internally rescaled for its own curve fitting.

Structure
---------
- Setup -- discover_smi_sessions, load_all_group_dfs_from_phase4 (both
  reimplemented unchanged from Phase 4/5).
- Function 6.0 -- identify_session_landmark_preference: per-session
  wrapper around identify_landmark_responses.
- Function 6.1 -- build_landmark_lookup_for_animal: loops 6.0 over every
  session for one animal.
- Function 6.2 -- merge_landmark_with_smi_table: attaches landmark
  columns onto Phase 4's saved per-cell comparison tables, by
  session_label + cell_idx.
- Composition diagnostics -- summarize_landmark_composition_by_layer /
  plot_landmark_composition_by_layer (per layer x condition landmark
  distribution + chi-square), run_landmark_composition_analysis_all_groups
  (driver across all 5 comparison groups).
- Population-level figures -- compute_population_landmark_fraction,
  plot_landmark_last_slope_population, plot_landmark_last_slope_by_layer
  -- the last-landmark-preference reduction under DCZ, population-level
  and per-layer.
- Open-loop vs. closed-loop diagnostic --
  summarize_openloop_vs_closedloop_effect -- effect size + per-layer
  direction reversal, closed-loop (DCZ1/2/3) vs. open-loop
  (Active_OL/Stationary_OL) groups.
- Option B -- build_all_sessions_landmark_smi_table,
  compare_smi_landmark1_vs_last, run_landmark1_vs_last_smi_analysis,
  plot_smi_landmark1_vs_last -- within-session (trial-count-free) test of
  whether landmark-4-preferring cells actually show higher SMI than
  landmark-1-preferring cells.
- Trial-count regression -- get_session_trial_count,
  test_condition_effect_on_landmark_metric, test_all_landmark_metrics --
  metric ~ n_trials + C(session_type), formally testing whether condition
  explains anything beyond recording length.
- Paired significance tests -- test_paired_significance_population,
  test_paired_significance_by_layer -- paired t-tests (population +
  per-layer) confirming the saline-vs-dcz pattern is statistically
  reproducible, not sampling noise.
- Cross-animal comparison -- load_paired_ttest_results_for_animal,
  compare_paired_ttest_across_animals -- loads each animal's already-saved
  paired t-test results rather than recomputing, side by side.
- Save-outputs -- save_dataframe_csv/save_figure_png/save_json,
  save_all_phase6_outputs, saved under Phase6_LandmarkPreference_Results/.

A GEE (cell-level, clustered-by-session) approach was tried and removed --
it sharpened the has_landmark_preference DCZ-vs-saline contrast to p=0.082
(JSY093, still not significant) but added complexity without changing the
overall picture; the paired t-test approach above is simpler and was kept
instead. Per-pair chi-square tests (treating cells as independent within
one session) were also tried and discarded -- they're pseudo-replicated
(hundreds of cells from one session are not independent replicates of
that condition) and produced misleadingly small p-values that don't
survive proper clustering.

Findings (JSY093, primary; JSY090 as the generalization check) -- NOT
conclusive
------------------------------------------------------------------------
1. The pattern is real and reproducible in JSY093. Landmark-4
   (reward-proximal, 120cm) preference is lower under DCZ than under its
   paired saline session in every one of 5 pairs, at the population level
   and in every individual layer. A paired t-test (the appropriately
   powered test here, since n=5 pairs is the correct independent unit)
   confirms this isn't sampling noise: population-level p=0.000855;
   per-layer p=0.0017-0.0200, all four layers significant, L5
   strongest/most consistent, L6 consistently weakest.
2. But "reproducible" does not mean "caused by DCZ." Saline sessions are
   the shortest recording of every pair (22-27 trials vs. 55-76 for DCZ,
   zero overlap), so trial count and condition are perfectly confounded
   WITHIN every pair -- pairing controls for what differs BETWEEN pairs,
   not this. The WLS regression built specifically to separate the two
   (metric ~ n_trials + C(session_type)) puts the direct DCZ-vs-saline
   contrast at p=0.279 for landmark-4 composition (frac_L4) -- not
   significant once trial count is in the model. The overall
   landmark-preferring rate (fraction_preferring) fares a little better --
   DCZ is significantly lower than a drug-free baseline even adjusting
   for trial count (p=0.002) -- but the direct DCZ-vs-saline contrast for
   that metric isn't significant either (p=0.220 by WLS; p=0.082 by a
   since-removed GEE model using full cell-level data, the closest
   anything in this phase came to significance on that specific contrast).
3. Layer pattern is general, not selective. All four layers show the
   reduction, not a subset concentrated in RSC's direct deep-layer
   targets (L5/L6) -- if this is a real DCZ effect, it isn't anatomically
   selective in the way the original hypothesis predicted. L6 is reliably
   the weakest layer across several independent diagnostics in this phase
   (DCZ1's non-significant composition chi-square, Active_OL's near-zero
   drop, the weakest paired-t p-value here too) -- a specific, recurring
   detail worth remembering, not noise.
4. Does not replicate in JSY090 -- the biggest caveat. The same
   closed-loop comparison (DCZ1/2/3) reverses direction in JSY090: 3-4 of
   4 layers flip (dcz > saline instead of saline > dcz), essentially a
   mirror image of JSY093's pattern. Confirmed with the same paired
   t-test used for JSY093: population-level p=0.463 (not significant, and
   the sign is even slightly negative), no layer significant, and
   same-direction agreement close to 50/50 in every layer rather than
   JSY093's 5/5 -- a genuine null, not an underpowered version of the same
   trend. One complication for a clean responder/non-responder
   interpretation: JSY093 is also the one animal where frac_L4 showed a
   significant trial-count relationship in the WLS regression (p=0.012);
   JSY090 showed none (p=0.619) -- so part of "JSY093 shows it, JSY090
   doesn't" could reflect JSY093's specific recordings being more
   trial-count-sensitive for this metric, not necessarily stronger
   biological RSC responsiveness. The two explanations aren't
   distinguishable from this comparison alone.
5. The original motivating premise -- landmark-4 preference indexes "more
   spatial" coding -- is not supported, and mildly contradicted. Option B
   (within-session, trial-count-free by construction) compared SMI
   between landmark-1- and landmark-4-preferring cells directly:
   landmark-1 cells show equal or HIGHER SMI, significant in JSY090
   (p=0.012, 8/9 sessions), directionally consistent in JSY093 (5/7
   sessions, p=0.219). A working alternative interpretation (not yet
   independently tested): landmark 1 and landmark 4 may encode different
   task-relevant information rather than differing in coding quality --
   landmark 1 as an early self-localization anchor, landmark 4 as a
   reward-proximity cue whose salience might build with experience. The
   clearest way to test that specifically -- within-session trial-block
   dynamics (analyze_within_session_dynamics in 3.LandmarkPreference,
   unused so far) -- was proposed and deferred, not run.
6. Bottom line: this phase produced a real, well-characterized,
   statistically robust phenomenon in JSY093 -- not an established
   DCZ/RSC finding. Its cause (drug vs. the trial-count confound baked
   into this dataset's design) remains genuinely undetermined, and it
   does not generalize to the one other animal tested. Resolving this
   needs either trial-count-matched recordings, more animals, or Track
   A's paired within-cell design (which sidesteps the population-level
   trial-count confound entirely, the way it does for Phases 3-5).

Correction (added later): the population-level metric used throughout
findings 1-4 above (frac_L{n} in the plots/paired tests) was originally
computed out of ALL cells (overall_frac_L{n} -- diluted by cells with no
landmark preference at all). It's now computed out of landmark-PREFERRING
cells only, which is the metric that actually answers "among cells that
show a landmark preference, which one" rather than conflating that with
"how many cells are landmark-tuned at all" (a separate question,
fraction_preferring already covers it). overall_frac_L{n} is still
computed and available for context, just no longer the default. The
specific numbers cited above (p=0.000855 etc.) were computed under the
old definition and are pending re-run under the new one -- the direction
and rough shape of the finding is expected to hold, but treat the exact
values as stale until re-confirmed. Separately, every population/layer
figure now plots closed-loop (DCZ1/2/3) and open-loop
(Active_OL/Stationary_OL) groups on separate figures rather than
overlaying all 5 -- they're different manipulations (normal VR vs.
self-motion decoupled from vision) and were always a different question,
even though the plots didn't visually distinguish them before.

Indexing / reuse convention (shared with Phases 0-5)
------------------------------------------------------
Every per-cell array here is indexed by POSITION in
stat[iscell[:, 0] == 1], exactly as in every earlier phase. No index
translation is ever needed between this phase's inputs (Phase 4's saved
{group}_comparison_table.csv + each session's raw preproc.h5) and its
outputs.
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation\3.LandmarkPreference")

import os
import re
import glob
import json
from collections import Counter

import numpy as np
import h5py
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # interactive popups need a real GUI backend
import matplotlib.pyplot as plt
from matplotlib import rcParams
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, wilcoxon, mannwhitneyu, ttest_rel

rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20

# Existing pipeline code, reused via import -- never modified.
from helper import files
from LandmarkPrefernce_SingleSessionAnalysis import identify_landmark_responses, plot_cells_by_landmark_assignment

# Landmark config for this cohort's VR track. Updated 2026-08-10 after the
# Function 6.0 diagnostic (plot_cells_by_landmark_assignment) showed the
# original [25,55,85,115]/{before:15-20,after:10} config picked up too much
# onset-adjacent activity in landmark 1's window -- matches the same swap
# made in 3.LandmarkPreference/LandmarkPrefernce_SingleSessionAnalysis.py's
# __main__ block.
LANDMARK_POSITIONS = [37, 65, 93, 120]
LANDMARK_WINDOWS_CONFIG = [
    {'before': 25, 'after': 0},  # landmark 1 at 37cm
    {'before': 25, 'after': 0},  # landmark 2 at 65cm
    {'before': 25, 'after': 0},  # landmark 3 at 93cm
    {'before': 25, 'after': 0},  # landmark 4 at 120cm
]


# =============================================================================
# Setup -- discover_smi_sessions (reimplemented unchanged from Phase 4)
# =============================================================================

def discover_smi_sessions(animal_dir):
    """
    Scan animal_dir for every already-computed *_smi_results_dreadd.h5
    file, labeling each by whichever known naming pattern its TSeries
    folder matches. Reimplemented unchanged from Phase 4's Function 4.2 --
    Phase 4's saved comparison CSVs don't carry the raw session path,
    so this discovery step is needed again here to map session_label ->
    tseries_dir.

    Parameters
    ----------
    animal_dir : str

    Returns
    -------
    catalog : dict
        {label: {'save_path': str, 'session_type': str, 'tseries_dir': str}}
    """
    save_paths = sorted(glob.glob(os.path.join(animal_dir, '**', '*_smi_results_dreadd.h5'),
                                   recursive=True))

    entries = []  # (base_label, session_type, save_path, tseries_dir, tseries_name)
    unmatched = []

    for save_path in save_paths:
        tseries_dir = os.path.dirname(save_path)
        tseries_name = os.path.basename(tseries_dir)
        parent_dir = os.path.dirname(tseries_dir)
        parent_name = os.path.basename(parent_dir)

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

        entries.append((base_label, session_type, save_path, tseries_dir, tseries_name))

    label_counts = Counter(e[0] for e in entries)

    catalog = {}
    for base_label, session_type, save_path, tseries_dir, tseries_name in entries:
        label = f'{base_label}__{tseries_name}' if label_counts[base_label] > 1 else base_label

        if label in catalog:
            print(f"WARNING: label '{label}' still collides after disambiguation -- "
                  f"keeping {catalog[label]['save_path']}, skipping {save_path}")
            continue

        catalog[label] = {
            'save_path': save_path,
            'session_type': session_type,
            'tseries_dir': tseries_dir,
        }

    print(f"Discovered {len(catalog)} sessions with saved SMI results under {animal_dir}:")
    for label, info in catalog.items():
        print(f"  [{info['session_type']:>8}] {label}  <-  {info['save_path']}")

    collided_labels = [l for l, c in label_counts.items() if c > 1]
    if collided_labels:
        print(f"\n{len(collided_labels)} label(s) had multiple TSeries and were disambiguated: "
              f"{collided_labels}")

    if unmatched:
        print(f"\n{len(unmatched)} session(s) didn't match a known naming pattern "
              f"(labeled 'unknown'): {unmatched}.")

    return catalog


def load_all_group_dfs_from_phase4(output_dir):
    """
    Load every comparison group's table already saved by Phase 4's
    Function 4.8 -- these already carry session_label, condition, layer,
    SMI, valid, cell_idx per row, which is exactly what Function 6.2
    merges the landmark columns onto.

    Parameters
    ----------
    output_dir : str
        e.g. os.path.join(ANIMAL_DIR, 'Phase4_SessionComparison_Results').

    Returns
    -------
    all_group_dfs : dict
        {group_name: df}.
    """
    csv_paths = sorted(glob.glob(os.path.join(output_dir, '*_comparison_table.csv')))

    if not csv_paths:
        raise FileNotFoundError(f"No *_comparison_table.csv found in {output_dir} -- "
                                 "has Phase 4's save step (Function 4.8) been run for this animal?")

    all_group_dfs = {}
    for csv_path in csv_paths:
        group_name = os.path.basename(csv_path)[:-len('_comparison_table.csv')]
        df = pd.read_csv(csv_path)
        all_group_dfs[group_name] = df
        print(f"Loaded '{group_name}': {len(df)} cell-rows <- {csv_path}")

    print(f"\nLoaded {len(all_group_dfs)} group(s) from {output_dir}: {list(all_group_dfs.keys())}")
    return all_group_dfs


# =============================================================================
# Function 6.0 -- identify_session_landmark_preference
# =============================================================================

def identify_session_landmark_preference(tseries_dir,
                                         landmark_positions=LANDMARK_POSITIONS,
                                         landmark_windows_config=LANDMARK_WINDOWS_CONFIG,
                                         landmark_window=10.0,
                                         boundary_exclusion=(5, 5),
                                         exclude_first_bins=5, exclude_last_bins=5,
                                         smoothing_sigma=1.0):
    """
    Per-session landmark-preference identification. Wraps
    identify_landmark_responses (imported unchanged from
    3.LandmarkPreference) around this session's own RAW preproc.h5 --
    NOT Phase 3's saved SMI-results h5, which stores an internally
    rescaled bin_centers for its own curve fitting.

    Parameters
    ----------
    tseries_dir : str
        A session's TSeries folder (contains *preproc*.h5).
    landmark_positions : list of float
    landmark_windows_config : list of dict
    landmark_window : float
        Fallback symmetric window, unused since landmark_windows_config is provided.
    boundary_exclusion : tuple of float
    exclude_first_bins, exclude_last_bins : int
    smoothing_sigma : float

    Returns
    -------
    landmark_df : pandas.DataFrame
        One row per cell (cell_idx = position in stat[iscell[:,0]==1]):
        has_landmark_preference, preferred_landmark_position, preference_strength.
    raw_results : dict
        Full identify_landmark_responses() return dict.
    """
    preproc_files = glob.glob(os.path.join(tseries_dir, "*preproc*.h5"))
    if not preproc_files:
        raise FileNotFoundError(f"No *preproc*.h5 found in {tseries_dir}")
    preproc_data = files.read_h5(preproc_files[0])

    normalized_spatial_activity = preproc_data['norm_spatial_activity']
    bin_centers = preproc_data['bin_centers']  # RAW cm scale

    raw_results = identify_landmark_responses(
        normalized_spatial_activity, bin_centers, landmark_positions,
        landmark_windows_config=landmark_windows_config,
        landmark_window=landmark_window,
        boundary_exclusion=boundary_exclusion,
        smoothing_sigma=smoothing_sigma,
        exclude_first_bins=exclude_first_bins,
        exclude_last_bins=exclude_last_bins,
    )

    n_cells = len(raw_results['valid_cells'])
    has_pref = raw_results['valid_cells']
    preferred_idx = raw_results['preferred_landmark']

    preferred_position = np.full(n_cells, np.nan)
    preferred_position[has_pref] = np.array(landmark_positions)[preferred_idx[has_pref]]

    landmark_df = pd.DataFrame({
        'cell_idx': np.arange(n_cells),
        'has_landmark_preference': has_pref,
        'preferred_landmark_position': preferred_position,
        'preference_strength': raw_results['preference_strength'],
    })

    return landmark_df, raw_results


# =============================================================================
# Function 6.1 -- build_landmark_lookup_for_animal
# =============================================================================

def build_landmark_lookup_for_animal(session_catalog, session_labels=None,
                                     landmark_positions=LANDMARK_POSITIONS,
                                     landmark_windows_config=LANDMARK_WINDOWS_CONFIG,
                                     landmark_window=10.0,
                                     boundary_exclusion=(5, 5),
                                     exclude_first_bins=5, exclude_last_bins=5,
                                     smoothing_sigma=1.0):
    """
    Loop identify_session_landmark_preference over every session needed,
    keyed by session_label so it lines up directly with Phase 4's saved
    comparison tables. Caches by session_label rather than recomputing
    per comparison group -- the same session (e.g. baseline Day5) can
    appear in multiple named comparison groups.

    Parameters
    ----------
    session_catalog : dict
        From discover_smi_sessions.
    session_labels : list of str, optional
        Defaults to every label in session_catalog.
    landmark_positions, landmark_windows_config, landmark_window,
    boundary_exclusion, exclude_first_bins, exclude_last_bins, smoothing_sigma :
        Passed through to identify_session_landmark_preference.

    Returns
    -------
    landmark_lookup : dict
        {session_label: landmark_df}.
    """
    if session_labels is None:
        session_labels = list(session_catalog.keys())

    landmark_lookup = {}
    for label in session_labels:
        info = session_catalog[label]
        print(f"\n--- {label} ---")
        landmark_df, _ = identify_session_landmark_preference(
            info['tseries_dir'],
            landmark_positions=landmark_positions,
            landmark_windows_config=landmark_windows_config,
            landmark_window=landmark_window,
            boundary_exclusion=boundary_exclusion,
            exclude_first_bins=exclude_first_bins,
            exclude_last_bins=exclude_last_bins,
            smoothing_sigma=smoothing_sigma,
        )
        landmark_lookup[label] = landmark_df
        n_pref = landmark_df['has_landmark_preference'].sum()
        print(f"  {label}: {n_pref}/{len(landmark_df)} cells with a landmark preference")

    print(f"\nBuilt landmark lookup for {len(landmark_lookup)} session(s).")
    return landmark_lookup


# =============================================================================
# Function 6.2 -- merge_landmark_with_smi_table
# =============================================================================

def merge_landmark_with_smi_table(comparison_df, landmark_lookup):
    """
    Attach landmark columns onto one of Phase 4's per-cell comparison
    tables, by session_label + cell_idx.

    Parameters
    ----------
    comparison_df : pandas.DataFrame
        One group's table (e.g. all_group_dfs['DCZ1']).
    landmark_lookup : dict
        {session_label: landmark_df} from Function 6.1.

    Returns
    -------
    merged_df : pandas.DataFrame
        comparison_df with has_landmark_preference/preferred_landmark_position/
        preference_strength attached. Rows whose session_label has no
        matching landmark_lookup entry get has_landmark_preference=False
        (flagged via a printed warning) rather than silently NaN.
    """
    missing_labels = set(comparison_df['session_label'].unique()) - set(landmark_lookup.keys())
    if missing_labels:
        print(f"WARNING: {len(missing_labels)} session_label(s) in comparison_df have no landmark "
              f"data in landmark_lookup -- their rows will get has_landmark_preference=False: "
              f"{missing_labels}")

    landmark_parts = []
    for label, landmark_df in landmark_lookup.items():
        part = landmark_df.copy()
        part['session_label'] = label
        landmark_parts.append(part)
    all_landmark_df = pd.concat(landmark_parts, ignore_index=True)

    merged_df = comparison_df.merge(all_landmark_df, on=['session_label', 'cell_idx'], how='left')

    n_missing = merged_df['has_landmark_preference'].isna().sum()
    if n_missing > 0:
        print(f"  {n_missing}/{len(merged_df)} row(s) have no landmark match after merge "
              f"(session_label not in landmark_lookup) -- set to has_landmark_preference=False.")
    merged_df['has_landmark_preference'] = merged_df['has_landmark_preference'].fillna(False).astype(bool)

    return merged_df


# =============================================================================
# Composition diagnostics -- summarize_landmark_composition_by_layer,
# plot_landmark_composition_by_layer, and the driver that runs both
# across every comparison group.
# =============================================================================

CANONICAL_LAYER_ORDER = ['L2/3', 'L4', 'L5', 'L6']


def _layer_order(layers_present):
    return ([l for l in CANONICAL_LAYER_ORDER if l in layers_present]
            + [l for l in layers_present if l not in CANONICAL_LAYER_ORDER])


def summarize_landmark_composition_by_layer(merged_df, layer_col='layer', condition_col='condition',
                                            preferred_landmark_col='preferred_landmark_position',
                                            has_pref_col='has_landmark_preference',
                                            landmark_positions=LANDMARK_POSITIONS):
    """
    Per layer x condition: n_cells, n_preferring, fraction_preferring, and
    (among preferring cells) the proportion assigned to each landmark. Plus
    a chi-square test per layer on the condition x landmark contingency
    table.

    Parameters
    ----------
    merged_df : pandas.DataFrame
        From merge_landmark_with_smi_table.
    layer_col, condition_col, preferred_landmark_col, has_pref_col : str
    landmark_positions : list of float

    Returns
    -------
    composition_df : pandas.DataFrame
        One row per (layer, condition).
    chi2_results : dict
        {layer: {'chi2', 'p', 'dof', 'contingency_table', 'conditions'} or None}.
    """
    layers_present = [l for l in merged_df[layer_col].dropna().unique()]
    layer_order = _layer_order(layers_present)

    rows = []
    chi2_results = {}

    for layer in layer_order:
        layer_df = merged_df[merged_df[layer_col] == layer]
        conditions_present = [c for c in layer_df[condition_col].dropna().unique()]

        contingency_rows = []
        for cond in conditions_present:
            cond_df = layer_df[layer_df[condition_col] == cond]
            n_cells = len(cond_df)
            preferring = cond_df[cond_df[has_pref_col]]
            n_preferring = len(preferring)

            row = {
                'layer': layer, 'condition': cond,
                'n_cells': n_cells, 'n_preferring': n_preferring,
                'fraction_preferring': n_preferring / n_cells if n_cells > 0 else np.nan,
            }
            landmark_counts = []
            for i, lm_pos in enumerate(landmark_positions):
                n_lm = int((preferring[preferred_landmark_col] == lm_pos).sum())
                row[f'frac_L{i+1}'] = n_lm / n_preferring if n_preferring > 0 else np.nan
                landmark_counts.append(n_lm)
            rows.append(row)
            contingency_rows.append(landmark_counts)

        valid_rows = [(cond, counts) for cond, counts in zip(conditions_present, contingency_rows)
                      if sum(counts) > 0]
        if len(valid_rows) >= 2:
            table = np.array([counts for _, counts in valid_rows])
            try:
                chi2, p, dof, _ = chi2_contingency(table)
                chi2_results[layer] = {'chi2': chi2, 'p': p, 'dof': dof,
                                       'contingency_table': table,
                                       'conditions': [c for c, _ in valid_rows]}
            except ValueError as e:
                print(f"  {layer}: chi-square skipped ({e})")
                chi2_results[layer] = None
        else:
            chi2_results[layer] = None

    composition_df = pd.DataFrame(rows)
    print(composition_df.to_string(index=False))

    print("\nChi-square (condition x landmark distribution, preferring cells only):")
    for layer, result in chi2_results.items():
        if result is not None:
            print(f"  {layer}: chi2={result['chi2']:.3f}, p={result['p']:.4f}, dof={result['dof']} "
                  f"(conditions: {result['conditions']})")
        else:
            print(f"  {layer}: skipped (not enough data)")

    return composition_df, chi2_results


def plot_landmark_composition_by_layer(composition_df, landmark_positions=LANDMARK_POSITIONS, title=''):
    """
    Grid of stacked bars (one panel per layer, one bar per condition --
    saline/dcz only, baseline excluded -- segments = landmark proportions
    among preferring cells).

    Each bar is annotated with Q1_pref/Q{n}_pref -- the population-level
    percentage of ALL cells (not just preferring ones) preferring the
    first/last landmark specifically (frac_L{i} * fraction_preferring).
    """
    layer_order = _layer_order(composition_df['layer'].unique())
    landmark_cols = [f'frac_L{i+1}' for i in range(len(landmark_positions))]
    colors = plt.cm.viridis(np.linspace(0, 1, len(landmark_positions)))
    condition_order = ['saline', 'dcz']
    n_landmarks = len(landmark_positions)

    fig, axes = plt.subplots(1, len(layer_order), figsize=(4.5 * len(layer_order), 6.5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, layer in zip(axes, layer_order):
        layer_rows = composition_df[composition_df['layer'] == layer]
        conditions = [c for c in condition_order if c in layer_rows['condition'].values]

        bottoms = np.zeros(len(conditions))
        for i, col in enumerate(landmark_cols):
            vals = np.array([layer_rows.loc[layer_rows['condition'] == c, col].values[0]
                             if c in layer_rows['condition'].values else 0 for c in conditions])
            vals = np.nan_to_num(vals)
            ax.bar(conditions, vals, bottom=bottoms, color=colors[i],
                   label=f'{landmark_positions[i]}cm' if ax is axes[0] else None)
            bottoms += vals

        for x, c in enumerate(conditions):
            row = layer_rows.loc[layer_rows['condition'] == c]
            if len(row) == 0:
                continue
            frac_pref = row['fraction_preferring'].values[0]
            n_cells = int(row['n_cells'].values[0])
            q1_pref = row['frac_L1'].values[0] * frac_pref
            q4_pref = row[f'frac_L{n_landmarks}'].values[0] * frac_pref
            ax.text(x, 1.03, f'Q1_pref={q1_pref:.0%}\nQ{n_landmarks}_pref={q4_pref:.0%}\n(n={n_cells})',
                   ha='center', fontsize=9)

        ax.set_ylim(0, 1.25)
        ax.set_title(layer)

    axes[0].set_ylabel('Proportion of preferring cells\nby landmark')
    axes[0].legend(fontsize=9, loc='upper left', bbox_to_anchor=(0, -0.08), ncol=len(landmark_positions))
    fig.suptitle(title, fontsize=18, fontweight='bold')
    plt.tight_layout()
    return fig


def run_landmark_composition_analysis_for_group(comparison_df, landmark_lookup, group_name=''):
    """
    Merge landmark data onto one group's table, then run
    summarize_landmark_composition_by_layer + plot_landmark_composition_by_layer.

    Parameters
    ----------
    comparison_df : pandas.DataFrame
        One group's table (e.g. all_group_dfs['DCZ2']).
    landmark_lookup : dict
        {session_label: landmark_df} from Function 6.1.
    group_name : str

    Returns
    -------
    result : dict
        {'merged_df', 'composition_df', 'chi2_results', 'fig'}.
    """
    print(f"\n{'='*90}\nLandmark composition analysis: {group_name}\n{'='*90}")

    merged_df = merge_landmark_with_smi_table(comparison_df, landmark_lookup)
    composition_df, chi2_results = summarize_landmark_composition_by_layer(merged_df)
    fig = plot_landmark_composition_by_layer(composition_df, title=f'{group_name} -- landmark composition by layer')
    plt.show()

    return {
        'merged_df': merged_df,
        'composition_df': composition_df,
        'chi2_results': chi2_results,
        'fig': fig,
    }


def run_landmark_composition_analysis_all_groups(all_group_dfs, landmark_lookup, condition_col='condition'):
    """
    Loop run_landmark_composition_analysis_for_group over every group,
    skipping single-condition groups (e.g. baseline -- no contrast to
    test).

    Parameters
    ----------
    all_group_dfs : dict
        {group_name: df}, from load_all_group_dfs_from_phase4.
    landmark_lookup : dict
        {session_label: landmark_df} from Function 6.1.
    condition_col : str

    Returns
    -------
    results : dict
        {group_name: run_landmark_composition_analysis_for_group(...) result}.
    """
    results = {}
    for group_name, df in all_group_dfs.items():
        conditions_present = df[condition_col].dropna().unique()
        if len(conditions_present) < 2:
            print(f"\n{'='*90}\n{group_name}: only {len(conditions_present)} condition(s) present "
                  f"({list(conditions_present)}) -- skipping (no contrast to test).\n{'='*90}")
            continue
        results[group_name] = run_landmark_composition_analysis_for_group(
            df, landmark_lookup, group_name=group_name)

    return results


# =============================================================================
# Population-level figures -- reduction in last-landmark preference under DCZ
# =============================================================================

# Fixed categorical color per comparison group -- never cycled/re-painted,
# same group always gets the same color across every figure in this module.
GROUP_COLORS = {
    'DCZ1': '#1b9e77',
    'DCZ2': '#d95f02',
    'DCZ3': '#7570b3',
    'Active_OL': '#e7298a',
    'Stationary_OL': '#66a61e',
}

# Closed-loop (regular saline/dcz, normal VR) vs. open-loop (self-motion
# decoupled from visual flow) groups -- different questions, so every
# population/layer figure below plots these separately rather than
# overlaying all 5 groups on one axis.
CLOSED_LOOP_GROUPS = ['DCZ1', 'DCZ2', 'DCZ3']
OPEN_LOOP_GROUPS = ['Active_OL', 'Stationary_OL']


def compute_population_landmark_fraction(landmark_composition_results, landmark_positions=LANDMARK_POSITIONS):
    """
    Population-level (all layers pooled) fraction of cells preferring
    each landmark, per group x condition -- two versions:
    frac_L1..frac_L{n} (out of PREFERRING cells only -- the primary
    metric: "among cells that show a landmark preference at all, which
    one") and overall_frac_L1..overall_frac_L{n} (out of ALL cells,
    kept as secondary/contextual information only -- diluting by
    non-preferring cells conflates "fewer cells are landmark-tuned at
    all" with "among tuned cells, the composition shifted," which are
    different questions; frac_L{n} is the one every plot/test in this
    module uses by default).

    Parameters
    ----------
    landmark_composition_results : dict
        {group_name: run_landmark_composition_analysis_for_group(...) result}.
    landmark_positions : list of float

    Returns
    -------
    pop_df : pandas.DataFrame
        One row per (group, condition): n_cells, n_preferring,
        fraction_preferring, frac_L1..frac_L{n} (out of preferring cells,
        primary), overall_frac_L1..overall_frac_L{n} (out of all cells,
        secondary).
    """
    rows = []
    for group_name, result in landmark_composition_results.items():
        merged_df = result['merged_df']
        for cond in merged_df['condition'].dropna().unique():
            cond_df = merged_df[merged_df['condition'] == cond]
            n_cells = len(cond_df)
            preferring = cond_df[cond_df['has_landmark_preference']]
            n_preferring = len(preferring)

            row = {
                'group': group_name, 'condition': cond,
                'n_cells': n_cells, 'n_preferring': n_preferring,
                'fraction_preferring': n_preferring / n_cells if n_cells > 0 else np.nan,
            }
            for i, lm_pos in enumerate(landmark_positions):
                n_lm = int((preferring['preferred_landmark_position'] == lm_pos).sum())
                row[f'frac_L{i+1}'] = n_lm / n_preferring if n_preferring > 0 else np.nan
                row[f'overall_frac_L{i+1}'] = n_lm / n_cells if n_cells > 0 else np.nan
            rows.append(row)

    pop_df = pd.DataFrame(rows)
    print(pop_df.to_string(index=False))
    return pop_df


def plot_landmark_last_slope_population(pop_df, landmark_positions=LANDMARK_POSITIONS,
                                        groups=None, subtitle=''):
    """
    Paired slope plot: one line per comparison group, saline -> dcz,
    population-level fraction of landmark-PREFERRING cells that prefer
    the last landmark (frac_L{n} -- out of preferring cells, not out of
    all cells; see compute_population_landmark_fraction). Baseline shown
    as a single horizontal reference line (all groups share the same
    baseline session).

    Parameters
    ----------
    pop_df : pandas.DataFrame
        From compute_population_landmark_fraction.
    landmark_positions : list of float
    groups : list of str, optional
        Which groups to plot -- e.g. CLOSED_LOOP_GROUPS or
        OPEN_LOOP_GROUPS, to keep the two kinds of comparison separate
        (they're different questions -- normal VR vs. self-motion
        decoupled from vision). Defaults to every group in pop_df.
    subtitle : str
        Appended to the title (e.g. 'closed-loop' / 'open-loop').

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    metric_col = f'frac_L{len(landmark_positions)}'
    if groups is None:
        groups = list(pop_df['group'].unique())
    plot_df = pop_df[pop_df['group'].isin(groups)]

    fig, ax = plt.subplots(figsize=(7.5, 8.5))

    baseline_val = pop_df.loc[pop_df['condition'] == 'baseline', metric_col].mean()
    ax.axhline(baseline_val, color='tab:blue', linestyle='--', linewidth=2, alpha=0.7,
               zorder=1, label=f'baseline ({baseline_val:.1%})')

    x_positions = {'saline': 0, 'dcz': 1}
    for group_name in groups:
        group_rows = plot_df[plot_df['group'] == group_name]
        color = GROUP_COLORS.get(group_name, 'gray')
        xs, ys = [], []
        for cond in ('saline', 'dcz'):
            val = group_rows.loc[group_rows['condition'] == cond, metric_col]
            if len(val) > 0:
                xs.append(x_positions[cond])
                ys.append(val.values[0])
        if len(xs) == 2:
            ax.plot(xs, ys, color=color, marker='o', markersize=11, linewidth=2.5,
                    solid_capstyle='round', zorder=3)
            ax.text(xs[-1] + 0.05, ys[-1], group_name, color=color, fontsize=14,
                    va='center', ha='left', fontweight='bold')

    ax.set_xlim(-0.3, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['saline', 'dcz'])
    ax.set_ylabel(f'Fraction of landmark-preferring cells\npreferring landmark {len(landmark_positions)} '
                  f'({landmark_positions[-1]:.0f}cm)')
    ax.legend(loc='upper right', fontsize=13, frameon=False)
    title = 'Last-landmark preference drops under DCZ'
    if subtitle:
        title += f'\n({subtitle})'
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_landmark_last_slope_by_layer(landmark_composition_results, landmark_positions=LANDMARK_POSITIONS,
                                      groups=None, subtitle=''):
    """
    Small multiples (one panel per layer) of the same paired slope plot,
    using composition_df's per-layer frac_L{n} directly -- fraction of
    landmark-PREFERRING cells in that layer preferring the last landmark,
    not diluted by non-preferring cells (see
    compute_population_landmark_fraction for why that distinction
    matters).

    Parameters
    ----------
    landmark_composition_results : dict
        {group_name: run_landmark_composition_analysis_for_group(...) result}.
    landmark_positions : list of float
    groups : list of str, optional
        Which groups to plot -- e.g. CLOSED_LOOP_GROUPS or
        OPEN_LOOP_GROUPS, kept separate since they're different
        questions. Defaults to every group in landmark_composition_results.
    subtitle : str
        Appended to the figure title (e.g. 'closed-loop' / 'open-loop').

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_landmarks = len(landmark_positions)
    last_col = f'frac_L{n_landmarks}'

    if groups is None:
        groups = list(landmark_composition_results.keys())
    plot_results = {g: landmark_composition_results[g] for g in groups
                    if g in landmark_composition_results}

    first_result = next(iter(plot_results.values()))
    layer_order = _layer_order(first_result['composition_df']['layer'].unique())

    fig, axes = plt.subplots(1, len(layer_order), figsize=(5 * len(layer_order), 7.5), sharey=True)
    axes = np.atleast_1d(axes)
    x_positions = {'saline': 0, 'dcz': 1}

    for ax, layer in zip(axes, layer_order):
        baseline_vals = []
        for group_name, result in plot_results.items():
            comp = result['composition_df']
            layer_rows = comp[comp['layer'] == layer].set_index('condition')

            if 'baseline' in layer_rows.index:
                baseline_vals.append(layer_rows.loc['baseline', last_col])

            color = GROUP_COLORS.get(group_name, 'gray')
            xs, ys = [], []
            for cond in ('saline', 'dcz'):
                if cond in layer_rows.index:
                    xs.append(x_positions[cond])
                    ys.append(layer_rows.loc[cond, last_col])
            if len(xs) == 2:
                ax.plot(xs, ys, color=color, marker='o', markersize=8, linewidth=2,
                        solid_capstyle='round', zorder=3)

        if baseline_vals:
            ax.axhline(np.mean(baseline_vals), color='tab:blue', linestyle='--',
                       linewidth=1.5, alpha=0.7, zorder=1)

        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['saline', 'dcz'])
        ax.set_title(layer)

    axes[0].set_ylabel(f'Fraction of preferring cells\npreferring landmark {n_landmarks} '
                       f'({landmark_positions[-1]:.0f}cm)')

    handles = [plt.Line2D([0], [0], color=GROUP_COLORS.get(g, 'gray'), marker='o', linewidth=2, label=g)
              for g in plot_results.keys()]
    handles.append(plt.Line2D([0], [0], color='tab:blue', linestyle='--', label='baseline'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=11,
              bbox_to_anchor=(0.5, -0.06), frameon=False)

    title = 'Last-landmark preference drop under DCZ, by layer'
    if subtitle:
        title += f' ({subtitle})'
    fig.suptitle(title)
    plt.tight_layout()
    return fig


# =============================================================================
# Open-loop vs. closed-loop diagnostic
# =============================================================================

def summarize_openloop_vs_closedloop_effect(landmark_composition_results, landmark_positions=LANDMARK_POSITIONS):
    """
    Population-level (per-layer, n_preferring-weighted) saline vs dcz
    far-landmark fraction per group -- out of landmark-PREFERRING cells,
    not out of all cells (see compute_population_landmark_fraction) --
    plus which layer(s), if any, reverse direction (dcz > saline).

    Parameters
    ----------
    landmark_composition_results : dict
        {group_name: run_landmark_composition_analysis_for_group(...) result}.
    landmark_positions : list of float

    Returns
    -------
    summary_df : pandas.DataFrame
        One row per group: saline_pop_frac_last, dcz_pop_frac_last,
        absolute_drop, relative_drop, reversed_layers.
    """
    last_col = f'frac_L{len(landmark_positions)}'
    rows = []

    for group_name, result in landmark_composition_results.items():
        comp = result['composition_df']
        pop = {}
        for cond in ('saline', 'dcz'):
            cond_rows = comp[comp['condition'] == cond]
            n_last_total = (cond_rows[last_col] * cond_rows['fraction_preferring'] * cond_rows['n_cells']).sum()
            n_preferring_total = (cond_rows['fraction_preferring'] * cond_rows['n_cells']).sum()
            pop[cond] = n_last_total / n_preferring_total if n_preferring_total > 0 else np.nan

        reversed_layers = []
        for layer in comp['layer'].unique():
            layer_rows = comp[comp['layer'] == layer].set_index('condition')
            if 'saline' in layer_rows.index and 'dcz' in layer_rows.index:
                if layer_rows.loc['dcz', last_col] > layer_rows.loc['saline', last_col]:
                    reversed_layers.append(layer)

        rows.append({
            'group': group_name,
            'saline_pop_frac_last': pop['saline'],
            'dcz_pop_frac_last': pop['dcz'],
            'absolute_drop': pop['saline'] - pop['dcz'],
            'relative_drop': (pop['saline'] - pop['dcz']) / pop['saline'] if pop['saline'] > 0 else np.nan,
            'reversed_layers': reversed_layers,
        })

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    return summary_df


# =============================================================================
# Option B -- does landmark-4 preference actually mean "more spatial"?
# Within-session (trial-count-free) SMI comparison, landmark-1- vs
# landmark-last-preferring cells.
# =============================================================================

def build_all_sessions_landmark_smi_table(landmark_composition_results):
    """
    Concatenate every group's merged_df and drop duplicate
    (session_label, cell_idx) rows -- since the shared baseline (Day5)
    appears identically in every group's merged_df, this collapses back
    down to one row per unique (session, cell) across all sessions
    discovered for this animal.

    Parameters
    ----------
    landmark_composition_results : dict
        {group_name: run_landmark_composition_analysis_for_group(...) result}.

    Returns
    -------
    all_sessions_df : pandas.DataFrame
    """
    all_df = pd.concat([r['merged_df'] for r in landmark_composition_results.values()], ignore_index=True)
    all_df = all_df.drop_duplicates(subset=['session_label', 'cell_idx']).reset_index(drop=True)
    print(f"{len(all_df)} cell-rows across {all_df['session_label'].nunique()} unique sessions.")
    print(all_df.groupby('session_label')['condition'].first().to_string())
    return all_df


def compare_smi_landmark1_vs_last(session_df, filter_col='valid',
                                  landmark_col='preferred_landmark_position', value_col='SMI',
                                  landmark_positions=LANDMARK_POSITIONS):
    """
    Within one session, Mann-Whitney U comparing SMI between cells
    preferring the first landmark vs cells preferring the last landmark
    only (not all 4 -- more power per group, and it's the direct
    near-vs-far contrast the hypothesis is about).

    Parameters
    ----------
    session_df : pandas.DataFrame
        One session's rows only (same trial count for every row).
    filter_col, landmark_col, value_col : str
    landmark_positions : list of float

    Returns
    -------
    result : dict or None
        None if either group has fewer than 2 cells. Otherwise:
        {'n_first', 'n_last', 'median_SMI_first', 'median_SMI_last',
         'U_stat', 'p'}.
    """
    first_lm, last_lm = landmark_positions[0], landmark_positions[-1]
    d = session_df[session_df[filter_col] & session_df['has_landmark_preference']
                   & session_df[landmark_col].isin([first_lm, last_lm])]

    smi_first = d.loc[d[landmark_col] == first_lm, value_col].to_numpy()
    smi_last = d.loc[d[landmark_col] == last_lm, value_col].to_numpy()

    if len(smi_first) < 2 or len(smi_last) < 2:
        return None

    u_stat, p = mannwhitneyu(smi_first, smi_last, alternative='two-sided')

    return {
        'n_first': len(smi_first), 'n_last': len(smi_last),
        'median_SMI_first': float(np.median(smi_first)),
        'median_SMI_last': float(np.median(smi_last)),
        'U_stat': u_stat, 'p': p,
    }


def run_landmark1_vs_last_smi_analysis(all_sessions_df):
    """
    Loops compare_smi_landmark1_vs_last over every session, builds a
    summary table, and a paired Wilcoxon test across sessions on the
    median-SMI difference (last - first) -- tests whether the direction
    is consistent regardless of condition.

    Parameters
    ----------
    all_sessions_df : pandas.DataFrame
        From build_all_sessions_landmark_smi_table.

    Returns
    -------
    results : dict
        {session_label: compare_smi_landmark1_vs_last(...) result or None}.
    summary_df : pandas.DataFrame
        One row per session with a valid comparison: session_label,
        condition, n_first, n_last, median_SMI_first, median_SMI_last,
        SMI_diff_last_minus_first, p.
    """
    results = {}
    summary_rows = []
    for label, session_df in all_sessions_df.groupby('session_label'):
        result = compare_smi_landmark1_vs_last(session_df)
        results[label] = result
        if result is None:
            print(f"{label}: skipped (fewer than 2 cells in landmark 1 or landmark-last group)")
            continue
        summary_rows.append({
            'session_label': label,
            'condition': session_df['condition'].iloc[0],
            'n_first': result['n_first'], 'n_last': result['n_last'],
            'median_SMI_first': result['median_SMI_first'],
            'median_SMI_last': result['median_SMI_last'],
            'SMI_diff_last_minus_first': result['median_SMI_last'] - result['median_SMI_first'],
            'p': result['p'],
        })

    summary_df = pd.DataFrame(summary_rows)
    print(f"\n{summary_df.to_string(index=False)}")

    n_positive = (summary_df['SMI_diff_last_minus_first'] > 0).sum()
    print(f"\n{n_positive}/{len(summary_df)} sessions show higher median SMI in "
          f"landmark-last-preferring cells than landmark-1-preferring cells "
          f"(regardless of condition).")

    if len(summary_df) >= 2:
        stat, p = wilcoxon(summary_df['median_SMI_first'], summary_df['median_SMI_last'])
        print(f"\nPaired Wilcoxon signed-rank across sessions (median_SMI_first vs "
              f"median_SMI_last, n={len(summary_df)} sessions): stat={stat:.3f}, p={p:.4f}")

    return results, summary_df


def plot_smi_landmark1_vs_last(summary_df, landmark_positions=LANDMARK_POSITIONS):
    """
    Slope plot: one line per session, x = landmark1-preferring vs
    landmark-last-preferring cells, y = median SMI within that session.
    Colored by condition (not group) -- the point here is whether the
    direction holds regardless of condition, since each line is already
    trial-count-free by construction (same session on both ends).

    Parameters
    ----------
    summary_df : pandas.DataFrame
        From run_landmark1_vs_last_smi_analysis.
    landmark_positions : list of float

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    color_by_condition = {'baseline': 'tab:blue', 'saline': 'tab:orange', 'dcz': 'tab:green'}
    fig, ax = plt.subplots(figsize=(7, 8.5))

    x_positions = {'first': 0, 'last': 1}
    for _, row in summary_df.iterrows():
        color = color_by_condition.get(row['condition'], 'gray')
        ax.plot([x_positions['first'], x_positions['last']],
                [row['median_SMI_first'], row['median_SMI_last']],
                color=color, marker='o', markersize=8, linewidth=1.8, alpha=0.75, zorder=3)

    handles = [plt.Line2D([0], [0], color=c, marker='o', linewidth=2, label=cond)
              for cond, c in color_by_condition.items() if cond in summary_df['condition'].values]
    ax.legend(handles=handles, loc='upper left', fontsize=13, frameon=False)

    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'landmark 1\n({landmark_positions[0]:.0f}cm)',
                        f'landmark {len(landmark_positions)}\n({landmark_positions[-1]:.0f}cm)'])
    ax.set_ylabel('Median SMI (within-session)')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_title('Within-session SMI: landmark-1- vs\nlandmark-last-preferring cells')
    plt.tight_layout()
    return fig


# =============================================================================
# Trial-count regression -- does condition explain anything beyond
# recording length? metric ~ n_trials + C(session_type), same design as
# Phase 4's Function 4.6.
# =============================================================================

def get_session_trial_count(tseries_dir):
    """
    Read a session's trial count straight from its preproc.h5. Reimplemented
    unchanged from Phase 4's Function 4.5.
    """
    preproc_files = glob.glob(os.path.join(tseries_dir, "*preproc*.h5"))
    if not preproc_files:
        raise FileNotFoundError(f"No *preproc*.h5 found in {tseries_dir}")
    with h5py.File(preproc_files[0], 'r') as f:
        n_trials = f['spatial_activity'].shape[1]
    return n_trials


def test_condition_effect_on_landmark_metric(calib_df, fraction_col, weight_col='n_cells',
                                             reference_level='baseline'):
    """
    Weighted least squares: fraction_col ~ n_trials + C(session_type),
    weighted by weight_col.

    Parameters
    ----------
    calib_df : pandas.DataFrame
        Needs fraction_col, n_trials, session_type, weight_col.
    fraction_col : str
    weight_col : str
    reference_level : str

    Returns
    -------
    model_result : statsmodels regression results object
    """
    df = calib_df.copy()
    other_levels = [c for c in df['session_type'].unique() if c != reference_level]
    df['session_type'] = pd.Categorical(df['session_type'], categories=[reference_level] + other_levels)

    formula = f"{fraction_col} ~ n_trials + C(session_type)"
    model_result = smf.wls(formula, data=df, weights=df[weight_col]).fit()

    print(f"\n=== {fraction_col} ~ n_trials + condition (weighted by {weight_col}, "
          f"reference='{reference_level}') ===")
    print(model_result.summary().tables[1])

    param_names = list(model_result.params.index)
    dcz_name = next((p for p in param_names if 'dcz' in p.lower()), None)
    saline_name = next((p for p in param_names if 'saline' in p.lower()), None)
    if dcz_name and saline_name:
        contrast = f"{dcz_name} - {saline_name}"
        print(f"\nDirect DCZ vs. saline contrast:")
        print(model_result.t_test(contrast))

    return model_result


def test_all_landmark_metrics(calib_df, reference_level='baseline'):
    """
    Runs test_condition_effect_on_landmark_metric for fraction_preferring
    (weighted by n_cells) and frac_L4 (weighted by n_preferring).

    Parameters
    ----------
    calib_df : pandas.DataFrame
    reference_level : str

    Returns
    -------
    results : dict
        {'fraction_preferring': model_result, 'frac_L4': model_result}.
    """
    results = {}
    results['fraction_preferring'] = test_condition_effect_on_landmark_metric(
        calib_df, 'fraction_preferring', weight_col='n_cells', reference_level=reference_level)
    results['frac_L4'] = test_condition_effect_on_landmark_metric(
        calib_df, 'frac_L4', weight_col='n_preferring', reference_level=reference_level)
    return results


# =============================================================================
# Paired significance tests -- population + per-layer. n=5 pairs is
# genuinely the correct independent unit; a paired t-test uses magnitude
# and consistency, not just sign, unlike the Wilcoxon signed-rank (which
# is capped at p=0.0625 at this n regardless of effect size).
# =============================================================================

def test_paired_significance_population(pop_df, metric_col=None, landmark_positions=LANDMARK_POSITIONS,
                                        groups=None):
    """
    Paired t-test (and Wilcoxon, for comparison) on the population-level
    metric across saline/dcz pairs.

    IMPORTANT: a significant result here confirms the pattern is real and
    reproducible across pairs -- it does NOT resolve whether it's
    DCZ-specific rather than trial-count-driven, since trial count differs
    systematically within every pair (saline always shorter) and this
    paired design can't separate the two. See
    test_condition_effect_on_landmark_metric for the test that actually
    controls for n_trials.

    Parameters
    ----------
    pop_df : pandas.DataFrame
        From compute_population_landmark_fraction.
    metric_col : str, optional
        Defaults to 'frac_L{n}' (the last landmark, out of preferring
        cells -- the primary metric).
    landmark_positions : list of float
    groups : list of str, optional
        Restrict to these groups (e.g. CLOSED_LOOP_GROUPS or
        OPEN_LOOP_GROUPS) -- defaults to every group in pop_df. Note:
        OPEN_LOOP_GROUPS only has 2 members, so a paired test there has
        just 1 degree of freedom -- barely informative on its own, mostly
        useful as a sanity check alongside the full-group result.

    Returns
    -------
    paired : pandas.DataFrame
        One row per group: saline, dcz, diff.
    t_stat, p_ttest : float
    """
    if metric_col is None:
        metric_col = f'frac_L{len(landmark_positions)}'

    df = pop_df if groups is None else pop_df[pop_df['group'].isin(groups)]
    saline_vals = df.loc[df['condition'] == 'saline'].set_index('group')[metric_col]
    dcz_vals = df.loc[df['condition'] == 'dcz'].set_index('group')[metric_col]
    paired = pd.DataFrame({'saline': saline_vals, 'dcz': dcz_vals}).dropna()
    paired['diff'] = paired['saline'] - paired['dcz']

    t_stat, p_ttest = ttest_rel(paired['saline'], paired['dcz'])
    w_stat, p_wilcoxon = wilcoxon(paired['saline'], paired['dcz'])

    print(f"Paired comparison on '{metric_col}' (n={len(paired)} pairs):")
    print(paired.to_string())
    print(f"\nPaired t-test: t={t_stat:.3f}, df={len(paired) - 1}, p={p_ttest:.6f}")
    print(f"Wilcoxon signed-rank (for comparison): stat={w_stat:.3f}, p={p_wilcoxon:.4f}")
    print("\nNOTE: significant here means the pattern is reproducible across pairs, NOT that it's")
    print("attributable to DCZ specifically rather than trial count.")

    return paired, t_stat, p_ttest


def test_paired_significance_by_layer(landmark_composition_results, landmark_positions=LANDMARK_POSITIONS,
                                      groups=None):
    """
    Same paired t-test, run separately per layer, using each group's
    composition_df's frac_L{n} directly -- fraction of preferring cells
    in that layer preferring the last landmark (out of preferring cells,
    not diluted by non-preferring cells).

    Parameters
    ----------
    landmark_composition_results : dict
        {group_name: run_landmark_composition_analysis_for_group(...) result}.
    landmark_positions : list of float
    groups : list of str, optional
        Restrict to these groups (e.g. CLOSED_LOOP_GROUPS or
        OPEN_LOOP_GROUPS) -- defaults to every group in
        landmark_composition_results. Same n=2/df=1 caveat as
        test_paired_significance_population applies to OPEN_LOOP_GROUPS.

    Returns
    -------
    summary_df : pandas.DataFrame
        One row per layer: n_pairs, mean_diff, std_diff, n_same_direction,
        paired_t_p, wilcoxon_p.
    layer_paired_data : dict
        {layer: DataFrame(group, saline, dcz)}.
    """
    last_col = f'frac_L{len(landmark_positions)}'
    if groups is None:
        groups = list(landmark_composition_results.keys())
    plot_results = {g: landmark_composition_results[g] for g in groups
                    if g in landmark_composition_results}

    first_result = next(iter(plot_results.values()))
    layer_order = _layer_order(first_result['composition_df']['layer'].unique())

    rows = []
    layer_paired_data = {}
    for layer in layer_order:
        saline_vals, dcz_vals, group_names = [], [], []
        for group_name, result in plot_results.items():
            comp = result['composition_df']
            layer_rows = comp[comp['layer'] == layer].set_index('condition')
            if 'saline' in layer_rows.index and 'dcz' in layer_rows.index:
                saline_vals.append(layer_rows.loc['saline', last_col])
                dcz_vals.append(layer_rows.loc['dcz', last_col])
                group_names.append(group_name)

        saline_vals = np.array(saline_vals)
        dcz_vals = np.array(dcz_vals)
        diffs = saline_vals - dcz_vals

        t_stat, p_ttest = ttest_rel(saline_vals, dcz_vals)
        w_stat, p_wilcoxon = wilcoxon(saline_vals, dcz_vals)

        layer_paired_data[layer] = pd.DataFrame(
            {'group': group_names, 'saline': saline_vals, 'dcz': dcz_vals})

        rows.append({
            'layer': layer, 'n_pairs': len(saline_vals),
            'mean_diff': diffs.mean(), 'std_diff': diffs.std(ddof=1),
            'n_same_direction': int((diffs > 0).sum()),
            'paired_t_p': p_ttest, 'wilcoxon_p': p_wilcoxon,
        })

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    return summary_df, layer_paired_data


# =============================================================================
# Cross-animal comparison -- loads each animal's already-saved paired
# t-test results (from save_all_phase6_outputs) rather than recomputing.
# =============================================================================

def load_paired_ttest_results_for_animal(animal_dir, animal_id):
    """
    Loads one animal's already-saved paired t-test results (from
    save_all_phase6_outputs), tagging each row with animal_id.

    Parameters
    ----------
    animal_dir : str
    animal_id : str

    Returns
    -------
    pop_df : pandas.DataFrame
        Population-level paired comparison (saline, dcz, diff), one row
        per group.
    layer_df : pandas.DataFrame
        Per-layer paired comparison summary.
    """
    output_dir = os.path.join(animal_dir, 'Phase6_LandmarkPreference_Results')

    pop_df = pd.read_csv(os.path.join(output_dir, 'paired_ttest_population.csv'), index_col=0)
    pop_df['animal_id'] = animal_id

    layer_df = pd.read_csv(os.path.join(output_dir, 'paired_ttest_by_layer.csv'))
    layer_df['animal_id'] = animal_id

    return pop_df, layer_df


def compare_paired_ttest_across_animals(animal_dirs):
    """
    Loads and combines every animal's saved paired t-test results
    (population + per-layer) into side-by-side comparison tables.

    Parameters
    ----------
    animal_dirs : dict
        {animal_id: animal_dir}.

    Returns
    -------
    combined_pop_df : pandas.DataFrame
    combined_layer_df : pandas.DataFrame
    """
    pop_parts, layer_parts = [], []
    for animal_id, animal_dir in animal_dirs.items():
        pop_df, layer_df = load_paired_ttest_results_for_animal(animal_dir, animal_id)
        pop_parts.append(pop_df)
        layer_parts.append(layer_df)

    combined_pop_df = pd.concat(pop_parts)
    combined_layer_df = pd.concat(layer_parts, ignore_index=True)

    print("Population-level paired t-test, by animal:")
    print(combined_pop_df.to_string())
    print("\nPer-layer paired t-test, by animal:")
    print(combined_layer_df.to_string(index=False))

    return combined_pop_df, combined_layer_df


# =============================================================================
# Save-outputs (standing rule) -- everything this phase generates gets
# saved before the phase is considered done.
# =============================================================================

def save_dataframe_csv(df, output_dir, filename, index=False):
    """
    Save a DataFrame to {output_dir}/{filename}, creating output_dir if
    needed. index=True for tables whose index is meaningful.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    df.to_csv(save_path, index=index)
    print(f"Saved -> {save_path}")
    return save_path


def save_figure_png(fig, output_dir, filename, dpi=150):
    """
    Save a matplotlib figure to {output_dir}/{filename}, creating
    output_dir if needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved -> {save_path}")
    return save_path


def _json_safe(obj):
    """Recursively convert numpy scalar types to native Python for json.dump."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json(data, output_dir, filename):
    """
    Save a JSON-serializable dict to {output_dir}/{filename}, creating
    output_dir if needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    with open(save_path, 'w') as f:
        json.dump(_json_safe(data), f, indent=2)
    print(f"Saved -> {save_path}")
    return save_path


def save_all_phase6_outputs(output_dir, landmark_composition_results, pop_df, openloop_effect_summary,
                            paired_population_df, paired_layer_summary_df, landmark_smi_summary_df,
                            calib_df, landmark_metric_test_results):
    """
    Saves everything Phase 6 generated for one animal. Figures are
    regenerated fresh from their underlying data rather than reused from
    whatever's currently in memory, so this is safe to call standalone.

    Saved under {ANIMAL_DIR}/Phase6_LandmarkPreference_Results/:
    - {group}_landmark_composition_by_layer.csv / .json (chi2) / _plot.png
      -- per comparison group.
    - population_landmark_fractions.csv,
      landmark_last_slope_population_closedloop.png,
      landmark_last_slope_population_openloop.png,
      landmark_last_slope_by_layer_closedloop.png,
      landmark_last_slope_by_layer_openloop.png -- closed-loop
      (DCZ1/2/3) and open-loop (Active_OL/Stationary_OL) plotted
      separately since they're different questions.
    - openloop_vs_closedloop_effect_summary.csv.
    - paired_ttest_population.csv, paired_ttest_by_layer.csv.
    - option_b_smi_landmark1_vs_last.csv, option_b_smi_landmark1_vs_last_plot.png.
    - trial_count_calibration.csv, wls_regression_{metric}.txt.

    Parameters
    ----------
    output_dir : str
        e.g. os.path.join(ANIMAL_DIR, 'Phase6_LandmarkPreference_Results').
    landmark_composition_results : dict
        {group_name: {'composition_df', 'chi2_results', ...}}.
    pop_df : pandas.DataFrame
    openloop_effect_summary : pandas.DataFrame
    paired_population_df : pandas.DataFrame
    paired_layer_summary_df : pandas.DataFrame
    landmark_smi_summary_df : pandas.DataFrame
    calib_df : pandas.DataFrame
    landmark_metric_test_results : dict
        {'fraction_preferring': model_result, 'frac_L4': model_result}.

    Returns
    -------
    saved_paths : dict
    """
    saved_paths = {}

    # 1. Per-group composition results
    for group_name, result in landmark_composition_results.items():
        saved_paths[f'{group_name}_composition'] = save_dataframe_csv(
            result['composition_df'], output_dir, f'{group_name}_landmark_composition_by_layer.csv')

        chi2_summary = {layer: ({'chi2': r['chi2'], 'p': r['p'], 'dof': r['dof'], 'conditions': r['conditions']}
                                if r is not None else None)
                        for layer, r in result['chi2_results'].items()}
        saved_paths[f'{group_name}_chi2'] = save_json(
            chi2_summary, output_dir, f'{group_name}_landmark_composition_chi2.json')

        fig = plot_landmark_composition_by_layer(
            result['composition_df'], title=f'{group_name} -- landmark composition by layer')
        saved_paths[f'{group_name}_composition_plot'] = save_figure_png(
            fig, output_dir, f'{group_name}_landmark_composition_by_layer_plot.png')
        plt.close(fig)

    # 2. Population-level summary + figures -- closed-loop and open-loop
    #    plotted separately (different questions: normal VR vs. self-motion
    #    decoupled from vision).
    saved_paths['pop_df'] = save_dataframe_csv(pop_df, output_dir, 'population_landmark_fractions.csv')

    for group_set, name in ((CLOSED_LOOP_GROUPS, 'closedloop'), (OPEN_LOOP_GROUPS, 'openloop')):
        present = [g for g in group_set if g in pop_df['group'].unique()]
        if not present:
            continue

        fig_pop = plot_landmark_last_slope_population(pop_df, groups=present, subtitle=name)
        saved_paths[f'slope_population_fig_{name}'] = save_figure_png(
            fig_pop, output_dir, f'landmark_last_slope_population_{name}.png')
        plt.close(fig_pop)

        present_in_results = [g for g in present if g in landmark_composition_results]
        if present_in_results:
            fig_layer = plot_landmark_last_slope_by_layer(
                landmark_composition_results, groups=present_in_results, subtitle=name)
            saved_paths[f'slope_layer_fig_{name}'] = save_figure_png(
                fig_layer, output_dir, f'landmark_last_slope_by_layer_{name}.png')
            plt.close(fig_layer)

    # 3. Open-loop vs closed-loop
    saved_paths['openloop_effect'] = save_dataframe_csv(
        openloop_effect_summary, output_dir, 'openloop_vs_closedloop_effect_summary.csv')

    # 4. Paired t-tests
    saved_paths['paired_population'] = save_dataframe_csv(
        paired_population_df, output_dir, 'paired_ttest_population.csv', index=True)
    saved_paths['paired_layer'] = save_dataframe_csv(
        paired_layer_summary_df, output_dir, 'paired_ttest_by_layer.csv')

    # 5. Option B
    saved_paths['option_b_summary'] = save_dataframe_csv(
        landmark_smi_summary_df, output_dir, 'option_b_smi_landmark1_vs_last.csv')
    smi_fig = plot_smi_landmark1_vs_last(landmark_smi_summary_df)
    saved_paths['option_b_fig'] = save_figure_png(
        smi_fig, output_dir, 'option_b_smi_landmark1_vs_last_plot.png')
    plt.close(smi_fig)

    # 6. Trial-count regression
    saved_paths['calib_df'] = save_dataframe_csv(calib_df, output_dir, 'trial_count_calibration.csv')

    os.makedirs(output_dir, exist_ok=True)
    for metric, model_result in landmark_metric_test_results.items():
        txt_path = os.path.join(output_dir, f'wls_regression_{metric}.txt')
        with open(txt_path, 'w') as f:
            f.write(str(model_result.summary()))
        print(f"Saved -> {txt_path}")
        saved_paths[f'wls_{metric}'] = txt_path

    print(f"\nSaved {len(saved_paths)} Phase 6 output(s) to {output_dir}")
    return saved_paths


# =============================================================================
# Driver
# =============================================================================

if __name__ == "__main__":
    ANIMAL_DIR = r"D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD"

    # --- Setup: discover sessions, build landmark lookup, load Phase 4's tables ---
    session_catalog = discover_smi_sessions(ANIMAL_DIR)
    landmark_lookup = build_landmark_lookup_for_animal(session_catalog)

    OUTPUT_DIR_PHASE4 = os.path.join(ANIMAL_DIR, 'Phase4_SessionComparison_Results')
    all_group_dfs = load_all_group_dfs_from_phase4(OUTPUT_DIR_PHASE4)

    # --- Composition analysis across all comparison groups ---
    landmark_composition_results = run_landmark_composition_analysis_all_groups(all_group_dfs, landmark_lookup)

    # --- Population-level figures -- closed-loop and open-loop kept
    #     separate since they're different questions ---
    pop_df = compute_population_landmark_fraction(landmark_composition_results)
    for group_set, name in ((CLOSED_LOOP_GROUPS, 'closed-loop'), (OPEN_LOOP_GROUPS, 'open-loop')):
        present = [g for g in group_set if g in pop_df['group'].unique()]
        if not present:
            continue
        fig_pop = plot_landmark_last_slope_population(pop_df, groups=present, subtitle=name)
        plt.show()
        present_in_results = [g for g in present if g in landmark_composition_results]
        if present_in_results:
            fig_layer = plot_landmark_last_slope_by_layer(
                landmark_composition_results, groups=present_in_results, subtitle=name)
            plt.show()

    # --- Open-loop vs. closed-loop diagnostic ---
    openloop_effect_summary = summarize_openloop_vs_closedloop_effect(landmark_composition_results)

    # --- Option B: within-session SMI, landmark-1 vs landmark-last ---
    all_sessions_df = build_all_sessions_landmark_smi_table(landmark_composition_results)
    landmark_smi_results, landmark_smi_summary_df = run_landmark1_vs_last_smi_analysis(all_sessions_df)
    smi_fig = plot_smi_landmark1_vs_last(landmark_smi_summary_df)
    plt.show()

    # --- Trial-count regression ---
    calib_rows = []
    for label, landmark_df in landmark_lookup.items():
        info = session_catalog[label]
        n_trials = get_session_trial_count(info['tseries_dir'])
        n_cells = len(landmark_df)
        preferring = landmark_df[landmark_df['has_landmark_preference']]
        n_preferring = len(preferring)
        frac_L4 = ((preferring['preferred_landmark_position'] == LANDMARK_POSITIONS[-1]).sum() / n_preferring
                  if n_preferring > 0 else np.nan)
        calib_rows.append({
            'session_label': label, 'session_type': info['session_type'], 'n_trials': n_trials,
            'n_cells': n_cells, 'n_preferring': n_preferring,
            'fraction_preferring': n_preferring / n_cells if n_cells > 0 else np.nan,
            'frac_L4': frac_L4,
        })
    calib_df = pd.DataFrame(calib_rows).sort_values('n_trials').reset_index(drop=True)
    landmark_metric_test_results = test_all_landmark_metrics(calib_df, reference_level='baseline')

    # --- Paired significance tests (population + per-layer) ---
    paired_population_df, pop_t_stat, pop_p_ttest = test_paired_significance_population(pop_df)
    paired_layer_summary_df, layer_paired_data = test_paired_significance_by_layer(landmark_composition_results)

    # --- Save everything ---
    PHASE6_OUTPUT_DIR = os.path.join(ANIMAL_DIR, 'Phase6_LandmarkPreference_Results')
    saved_paths = save_all_phase6_outputs(
        PHASE6_OUTPUT_DIR, landmark_composition_results, pop_df, openloop_effect_summary,
        paired_population_df, paired_layer_summary_df, landmark_smi_summary_df,
        calib_df, landmark_metric_test_results)

    # --- Optional: once this has been run for every animal (each with its own
    #     ANIMAL_DIR above), compare across animals using their saved outputs ---
    # ANIMAL_DIRS_PHASE6 = {
    #     'JSY093': r"D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD",
    #     'JSY090': r"D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD",
    # }
    # combined_pop_paired_df, combined_layer_paired_df = compare_paired_ttest_across_animals(ANIMAL_DIRS_PHASE6)
