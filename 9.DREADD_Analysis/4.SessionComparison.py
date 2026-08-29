"""
Phase 4 -- Session-level SMI Comparisons (DREADD saline/DCZ cohort)
=====================================================================

Consolidated from 4.SessionComparison.ipynb once every function below
worked end-to-end on real JSY090 and JSY093 data.

Scope
-----
Both tracks now, neither recomputes SMI -- both only load and compare what
Phase 3 already produced:
- **Track B** (population-level, no cross-session cell tracking) --
  Functions 4.1-4.8, reusing Phase 3's saved `*_smi_results_dreadd.h5` per
  session.
- **Track A** (tracked-cell, paired within-cell) -- Functions 4.9-4.13,
  reusing Phase 3's saved `{group}_track_a_smi.csv` per tracking group
  (Function 3.6/`load_track_a_group`'s joined output). Active as of
  JSY093's DCZ1/DCZ2/DCZ3 groups, run separately (never pooled). Track A
  has NOT yet been added to Phase 5 (layer-specific) or Phase 6 (landmark
  preference) -- that remains the next body of work.

Comparison groups used so far (named per-animal via Function 4.4's
interactive picker, not hardcoded -- exact session labels differ animal to
animal):
- `DCZ1`/`DCZ2`/`DCZ3` -- baseline (Day5) + each within-day saline/DCZ pair
- `Active_OL` / `Stationary_OL` -- baseline (Day5) + each open-loop
  saline/DCZ pair
- `baseline` -- all 5 baseline days pooled (single condition -- see
  Function 4.7's fallback for how this one gets compared)

Findings so far (both animals, Track B, single-session-per-condition)
-----------------------------------------------------------------------
- Reliability-fraction calibration (Functions 4.5/4.6) on JSY090: saline
  sessions' lower `combined_reliable` fraction is largely, but not
  entirely, explained by their shorter trial counts; DCZ itself was
  statistically indistinguishable from baseline once trial count was
  controlled for. Decomposing into `reliable_cells`/`pattern_reliable`/
  `active` components pointed to saline sessions specifically sitting
  below the trial-count trend, not DCZ sitting above it -- but note the
  saline trial-count range (10-28) barely overlaps baseline/DCZ's (40-118),
  so this regression is mostly extrapolating, not interpolating; treat
  cautiously. Reliability (reproducibility) is NOT the same thing as
  spatial selectivity (SMI) -- Function 3.3 already showed these can
  dissociate (`high_smi_but_unreliable` / `reliable_but_low_smi` cells
  exist), so none of this speaks to spatial coding directly.
- The actual hypothesis test (Function 4.7, SMI itself, `valid`-filtered):
  JSY090 showed NO significant DCZ-vs-saline difference in any of 5
  within-day/open-loop groups (though numerically lower under DCZ in 4/5,
  a consistent but individually-underpowered trend). JSY093 showed a
  SIGNIFICANT DCZ-vs-saline difference (DCZ lower) in 4 of 5 groups, with
  one large effect (DCZ2: median SMI 0.68 -> 0.38). This is a real
  animal-to-animal difference (n=2 so far) -- plausibly a
  responder/non-responder split common in chemogenetic experiments
  (DREADD expression/placement can vary substantially animal to animal) --
  not something to average away. Check histology/expression data for these
  two animals if available. More animals are needed before any
  population-level conclusion.

IMPORTANT -- Track B vs. Track A's reliability filter
---------------------------------------------------------------------
(Same note as in 3.SMICalculation.py's module docstring -- duplicated here
since Function 4.7 is where it bites hardest.) Track B has no
cross-session cell identity, so a Track B comparison can never restrict to
"cells reliable in both sessions" using each session's own full data --
every Track B comparison re-establishes "reliable"/"valid" independently
per session, at that session's own trial count. This can bias both the
reliable-cell fraction and the SMI magnitude among survivors (shorter/
noisier sessions show survivorship bias toward only the most
robustly-tuned cells). Track A's paired within-cell comparison (Function
4.10) sidesteps this entirely, since the same tracked cell's SMI is
compared across sessions regardless of either session's trial count -- it
does NOT need trial-matching to be valid, unlike Track B. Treat Track A as
the more decisive test of the saline-vs-DCZ question, not just a "more
sensitive" alternative to Track B -- though note Track A's OWN N is still
trial-count-sensitive in a different way, since a cell must clear the
(non-duration-adjusted) reliability bar in BOTH sessions to enter the
paired comparison at all, and the shorter saline session clears far fewer
cells (see 3.SMICalculation.py's run output on JSY093: 101/697, 32/490,
57/594 cells reliable in both sessions across DCZ1/DCZ2/DCZ3 -- decided to
proceed with this filter as-is rather than loosen it, matching Track B's
own precedent of trying and rejecting an alternative reliability
threshold).

Indexing convention (shared with Phases 0/1/2/3)
--------------------------------------------------
Every per-cell array here is indexed by POSITION in
`stat[iscell[:, 0] == 1]`, exactly as in every earlier phase. No index
translation is ever needed between this phase's inputs (Phase 3's saved
`*_smi_results_dreadd.h5`) and its outputs.
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import glob
import json
from collections import Counter
from itertools import combinations

import numpy as np
import h5py
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # interactive popups (checkbox pickers etc.) need a real GUI backend
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.widgets import CheckButtons, Button
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import kruskal, mannwhitneyu, wilcoxon, ttest_rel

rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20

# Existing pipeline code, reused via import -- never modified. Needed for
# Function 4.6's reliability-component recomputation (pattern_reliable /
# active_cells), same functions Phase 3's Function 3.3b already used.
from helper import files
from helper.ReliabilityTesting import evaluate_pattern_similarity_improved, improved_activity_threshold_check


# =============================================================================
# Function 4.1 -- load one session's already-computed SMI output
# =============================================================================

def load_session_smi_for_comparison(save_path, condition_label=None, session_label=None):
    """
    Load one session's already-computed *_smi_results_dreadd.h5 (Function
    3.2's output) directly from disk into a tidy per-cell DataFrame -- the
    shared building block every comparison in this phase starts from.
    Reading from disk rather than reusing an in-memory
    run_smi_analysis_session result decouples this phase from Phase 3's
    kernel session, and means Track A's later revisit can reuse this same
    loader unchanged.

    Parameters
    ----------
    save_path : str
        Path to a *_smi_results_dreadd.h5 file (Function 3.2's output).
    condition_label : str, optional
        Tag stored in a 'condition' column. Leave None to fill in later
        (e.g. via build_comparison_table).
    session_label : str, optional
        Tag stored in a 'session_label' column. Defaults to the h5's
        stored session_label attr.

    Returns
    -------
    df : pandas.DataFrame
        One row per cell: cell_idx, SMI, valid, analysis_reliable,
        combined_reliable, avg_cc, cohen_d, layer, condition, session_label.
        'valid' is already reliable_valid_cells (reliable AND the SMI
        curve fit succeeded) -- the exact mask run_smi_analysis_session
        itself uses for layer-level SMI stats, so no extra AND-ing is
        needed before comparing SMI values.
    """
    with h5py.File(save_path, 'r') as f:
        stored_label = f.attrs.get('session_label', os.path.basename(save_path))
        SMI_values = f['global_smi/SMI_all_cells'][:]
        valid_cells_mask = f['global_smi/valid_cells_mask'][:]
        analysis_reliable_cells = f['global_smi/analysis_reliable_cells'][:]
        combined_reliable = f['reliability/combined_reliable'][:]
        avg_cc = f['reliability/avg_cc'][:]
        cohen_d = f['reliability/cohen_d'][:]

        n_cells = len(SMI_values)
        layer_of_cell = np.full(n_cells, None, dtype=object)
        for safe_name in f['layer_smi']:
            layer_grp = f['layer_smi'][safe_name]
            original_name = layer_grp.attrs.get('original_name', safe_name)
            cell_indices = layer_grp['cell_indices'][:]
            layer_of_cell[cell_indices] = original_name

    df = pd.DataFrame({
        'cell_idx': np.arange(n_cells),
        'SMI': SMI_values,
        'valid': valid_cells_mask,
        'analysis_reliable': analysis_reliable_cells,
        'combined_reliable': combined_reliable,
        'avg_cc': avg_cc,
        'cohen_d': cohen_d,
        'layer': layer_of_cell,
    })
    df['condition'] = condition_label
    df['session_label'] = session_label if session_label is not None else stored_label

    print(f"Loaded {n_cells} cells from {save_path}")
    print(f"  analysis_reliable: {df['analysis_reliable'].sum()}, valid: {df['valid'].sum()}")

    return df


# =============================================================================
# Function 4.2 -- discover already-computed SMI sessions + checkbox picker
# =============================================================================

def discover_smi_sessions(animal_dir):
    """
    Scan animal_dir for every already-computed *_smi_results_dreadd.h5
    file, labeling each by whichever known naming pattern its TSeries
    folder matches. Phase 4 only ever works from Phase 3's saved output,
    never recomputes SMI itself.

    Nothing is silently dropped on a label collision -- every colliding
    entry is disambiguated by appending its raw TSeries folder name, so
    all of them show up in the picker and you choose which one(s) belong
    (same disambiguation behavior as Phase 3's discover_animal_sessions).

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
              f"(labeled 'unknown'): {unmatched}. If you know what one of these actually is, "
              f"the cleanest fix is to rename its TSeries folder to include SAL/DCZ/DayN; "
              f"otherwise just remember its true condition when you select it in the picker.")

    return catalog


def select_sessions_popup(session_catalog,
                           title='Select sessions to include in this comparison'):
    """
    Checkbox popup listing every session in session_catalog. Identical to
    Phase 3's function of the same name -- only reads session_type, so it
    works unchanged regardless of whether entries carry a plane0_path or a
    save_path.

    Parameters
    ----------
    session_catalog : dict
    title : str

    Returns
    -------
    selected_labels : list of str
    """
    labels = list(session_catalog.keys())
    n = len(labels)
    display_labels = [f"[{session_catalog[l]['session_type']:>8}] {l}" for l in labels]

    fig_height = max(4, 0.35 * n + 1.5)
    fig = plt.figure(figsize=(9, fig_height))
    try:
        fig.canvas.manager.set_window_title('SELECT SESSIONS -- check boxes, then Confirm (or Enter)')
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


# =============================================================================
# Function 4.3 -- build one comparison group's tidy table
# =============================================================================

def build_comparison_table(session_catalog, selected_labels):
    """
    Load every selected session via load_session_smi_for_comparison and
    concatenate into one tidy long-format DataFrame. condition is pulled
    straight from the catalog's session_type ('baseline'/'saline'/'dcz')
    -- no manual mapping needed, since which specific sessions you check
    in the picker is exactly what defines a given named comparison.

    Parameters
    ----------
    session_catalog : dict
        From discover_smi_sessions.
    selected_labels : list of str
        From select_sessions_popup (or any list of labels you already have).

    Returns
    -------
    df : pandas.DataFrame
    """
    if len(selected_labels) == 0:
        raise ValueError("selected_labels is empty -- nothing to build a comparison table from.")

    session_dfs = []
    for label in selected_labels:
        info = session_catalog[label]
        session_df = load_session_smi_for_comparison(
            info['save_path'], condition_label=info['session_type'], session_label=label
        )
        session_dfs.append(session_df)

    df = pd.concat(session_dfs, ignore_index=True)

    print(f"\nCombined comparison table: {len(df)} cell-rows across {len(selected_labels)} sessions")
    print(df.groupby('condition')['analysis_reliable'].agg(['sum', 'count']))

    return df


# =============================================================================
# Function 4.4 -- manual multi-group loop: name -> pick sessions -> "more?" -> repeat
# =============================================================================

def ask_more_groups_popup():
    """
    Yes/No popup: "Add another comparison group?"

    Returns
    -------
    bool
        True to define another group, False to stop. Closing the window
        without clicking either button counts as False (stop), so the
        loop can never hang waiting on a window that's gone.
    """
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis('off')
    fig.suptitle('Add another comparison group?', fontsize=14, fontweight='bold')
    try:
        fig.canvas.manager.set_window_title('MORE GROUPS?')
    except Exception:
        pass

    yes_ax = fig.add_axes([0.12, 0.15, 0.35, 0.35])
    no_ax = fig.add_axes([0.53, 0.15, 0.35, 0.35])
    yes_button = Button(yes_ax, 'Yes, add another')
    no_button = Button(no_ax, 'No, done')

    state = {'choice': None}

    def on_yes(event=None):
        state['choice'] = True
        fig.canvas.stop_event_loop()

    def on_no(event=None):
        state['choice'] = False
        fig.canvas.stop_event_loop()

    yes_button.on_clicked(on_yes)
    no_button.on_clicked(on_no)

    plt.show(block=False)
    while state['choice'] is None and plt.fignum_exists(fig.number):
        fig.canvas.start_event_loop(0.1)

    if plt.fignum_exists(fig.number):
        plt.close(fig)
        plt.pause(0.01)

    if state['choice'] is None:
        print("Window closed without choosing -- treating as 'No, done'.")
        return False

    return state['choice']


def collect_comparison_groups_interactively(session_catalog):
    """
    Loop: type a group name -> pick its sessions -> "more groups?" popup ->
    repeat or stop.

    Parameters
    ----------
    session_catalog : dict
        From discover_smi_sessions.

    Returns
    -------
    comparison_groups : dict
        {group_name: selected_labels}, in the order you defined them.
    """
    comparison_groups = {}

    while True:
        group_name = input("Name for this comparison group (e.g. 'DCZ1'): ").strip()
        if not group_name:
            print("Empty name -- skipping this group.")
        elif group_name in comparison_groups:
            print(f"'{group_name}' was already defined -- skipping (pick a different name to redo it).")
        else:
            selected_labels = select_sessions_popup(
                session_catalog, title=f"Select sessions for comparison group '{group_name}'"
            )
            if len(selected_labels) == 0:
                print(f"No sessions selected for '{group_name}' -- not adding this group.")
            else:
                comparison_groups[group_name] = selected_labels
                print(f"Added group '{group_name}': {selected_labels}")

        if not ask_more_groups_popup():
            break

    print(f"\nDefined {len(comparison_groups)} comparison group(s): {list(comparison_groups.keys())}")
    return comparison_groups


def run_all_comparisons(session_catalog, comparison_groups):
    """
    Loop build_comparison_table over every manually-defined group.

    Parameters
    ----------
    session_catalog : dict
        From discover_smi_sessions.
    comparison_groups : dict
        From collect_comparison_groups_interactively (or any
        {group_name: [labels]} dict you already have).

    Returns
    -------
    all_group_dfs : dict
        {group_name: df} -- df is build_comparison_table's output for
        that group.
    """
    all_group_dfs = {}

    for group_name, selected_labels in comparison_groups.items():
        print(f"\n{'='*90}\nComparison group: {group_name}\n{'='*90}")
        all_group_dfs[group_name] = build_comparison_table(session_catalog, selected_labels)

    print(f"\n{'='*90}\nAll {len(all_group_dfs)} comparison group(s) built: {list(all_group_dfs.keys())}")

    return all_group_dfs


# =============================================================================
# Function 4.5 -- reliability-fraction vs. trial-count calibration check
#
# Cheap diagnostic (no reruns): is the saline-vs-DCZ reliable-fraction gap
# explained by saline's shorter recordings, or is something else going on?
# Findings on JSY090 are summarized in the module docstring above.
# =============================================================================

def get_session_trial_count(tseries_dir):
    """
    Read a session's trial count straight from its preproc.h5, without
    loading the full spatial_activity array.

    Parameters
    ----------
    tseries_dir : str
        A session's TSeries folder (contains *preproc*.h5).

    Returns
    -------
    n_trials : int
    """
    preproc_files = glob.glob(os.path.join(tseries_dir, "*preproc*.h5"))
    if not preproc_files:
        raise FileNotFoundError(f"No *preproc*.h5 found in {tseries_dir}")
    with h5py.File(preproc_files[0], 'r') as f:
        n_trials = f['spatial_activity'].shape[1]
    return n_trials


def build_reliability_trial_count_calibration(session_catalog):
    """
    For every session with saved SMI results, get (n_trials, reliable_fraction)
    and fit a simple linear trend -- a cheap way to check whether a
    session's reliable-cell fraction is "expected" given how many trials
    it had, or unusually low/high relative to other sessions of similar N.

    Parameters
    ----------
    session_catalog : dict
        From discover_smi_sessions.

    Returns
    -------
    calib_df : pandas.DataFrame
        One row per session: session_label, session_type, n_trials,
        n_cells, n_reliable, reliable_fraction, predicted_fraction,
        residual.
    coeffs : numpy.ndarray
        [slope, intercept] of the fitted line (reliable_fraction ~ n_trials).
    """
    rows = []
    skipped = []
    for label, info in session_catalog.items():
        try:
            n_trials = get_session_trial_count(info['tseries_dir'])
        except FileNotFoundError:
            skipped.append(label)
            continue

        session_df = load_session_smi_for_comparison(info['save_path'], session_label=label)
        n_cells = len(session_df)
        n_reliable = int(session_df['analysis_reliable'].sum())
        rows.append({
            'session_label': label,
            'session_type': info['session_type'],
            'n_trials': n_trials,
            'n_cells': n_cells,
            'n_reliable': n_reliable,
            'reliable_fraction': n_reliable / n_cells if n_cells > 0 else np.nan,
        })

    if skipped:
        print(f"WARNING: {len(skipped)} session(s) skipped -- no *preproc*.h5 found on disk "
              f"(needed for trial count, separate from the already-saved SMI results): {skipped}")

    calib_df = pd.DataFrame(rows)

    coeffs = np.polyfit(calib_df['n_trials'], calib_df['reliable_fraction'], deg=1)
    calib_df['predicted_fraction'] = np.polyval(coeffs, calib_df['n_trials'])
    calib_df['residual'] = calib_df['reliable_fraction'] - calib_df['predicted_fraction']

    print(f"Fitted trend: reliable_fraction ~ {coeffs[0]:.6f} * n_trials + {coeffs[1]:.4f}")
    print("\nSorted by residual (most below-trend first -- worth a closer look):")
    print(calib_df.sort_values('residual')[
        ['session_label', 'session_type', 'n_trials', 'reliable_fraction', 'residual']
    ].to_string(index=False))

    return calib_df, coeffs


def plot_reliability_calibration(calib_df, coeffs):
    """
    Scatter of n_trials vs. reliable_fraction, colored by session_type,
    with the fitted trend line overlaid.

    Parameters
    ----------
    calib_df : pandas.DataFrame
        From build_reliability_trial_count_calibration.
    coeffs : numpy.ndarray
        From build_reliability_trial_count_calibration.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    color_by_type = {'baseline': 'tab:blue', 'saline': 'tab:orange', 'dcz': 'tab:green'}

    fig, ax = plt.subplots(figsize=(9, 8))
    for session_type, group in calib_df.groupby('session_type'):
        ax.scatter(group['n_trials'], group['reliable_fraction'],
                   c=color_by_type.get(session_type, 'gray'), s=80, alpha=0.8,
                   label=session_type)

    x_line = np.linspace(calib_df['n_trials'].min(), calib_df['n_trials'].max(), 100)
    ax.plot(x_line, np.polyval(coeffs, x_line), color='black', linestyle='--',
            label='linear fit')

    ax.set_xlabel('Trial count')
    ax.set_ylabel('Reliable fraction (analysis_reliable)')
    ax.set_title('Reliable-cell fraction vs. trial count')
    ax.legend(loc='best')

    plt.tight_layout()
    return fig


# =============================================================================
# Function 4.6 -- decompose reliability into its components, test each for a
# condition effect beyond trial count
#
# combined_reliable = reliable_cells & pattern_reliable & active_cells.
# reliable_cells uses a proxy (avg_cc/cohen_d floors, already saved, free)
# rather than a full shuffle rerun -- justified since Function 3.3b found
# the shuffle-significance step excluded zero additional cells beyond
# those floors on the Day1 test session. pattern_reliable/active_cells
# need spatial_activity reloaded (cheap I/O) but not shuffled.
# =============================================================================

def compute_reliability_components_for_session(tseries_dir, avg_cc, cohen_d,
                                                min_cc_threshold=0.1, cohen_threshold=0.8,
                                                min_pattern_corr=0.3, peak_distance_threshold=5,
                                                activity_method='absolute_percentile'):
    """
    Per-session reliable_cells/pattern_reliable/active fractions. See
    module-level comment above for why reliable_cells uses a proxy rather
    than a full rerun.

    Parameters
    ----------
    tseries_dir : str
        A session's TSeries folder (contains *preproc*.h5).
    avg_cc, cohen_d : numpy.ndarray
        Already-saved per-cell values (e.g. from load_session_smi_for_comparison).
    min_cc_threshold, cohen_threshold, min_pattern_corr, peak_distance_threshold : float
        Must match Preprocess.py's actual call for this session's area
        (same caveat as Function 3.3b).
    activity_method : str

    Returns
    -------
    dict with 'reliable_cells_fraction', 'pattern_reliable_fraction', 'active_fraction'.
    """
    preproc_files = glob.glob(os.path.join(tseries_dir, "*preproc*.h5"))
    if not preproc_files:
        raise FileNotFoundError(f"No *preproc*.h5 found in {tseries_dir}")
    preproc_data = files.read_h5(preproc_files[0])
    spatial_activity = preproc_data['spatial_activity']

    n_cells = spatial_activity.shape[0]

    active_cells, _ = improved_activity_threshold_check(spatial_activity, method=activity_method)
    pattern_reliable, odd_even_corr, peak_distances = evaluate_pattern_similarity_improved(
        spatial_activity, min_pattern_corr, peak_distance_threshold
    )

    reliable_cells_proxy = (avg_cc > min_cc_threshold) & (cohen_d > cohen_threshold)

    return {
        'reliable_cells_fraction': float(np.sum(reliable_cells_proxy)) / n_cells,
        'pattern_reliable_fraction': float(np.sum(pattern_reliable)) / n_cells,
        'active_fraction': float(np.sum(active_cells)) / n_cells,
    }


def build_reliability_component_calibration(session_catalog,
                                             min_cc_threshold=0.1, cohen_threshold=0.8,
                                             min_pattern_corr=0.3, peak_distance_threshold=5,
                                             activity_method='absolute_percentile'):
    """
    One table with combined_reliable plus its three components
    (reliable_cells proxy, pattern_reliable, active) per session, alongside
    n_trials/session_type/n_cells. Uses the true combined_reliable (not
    analysis_reliable), so the three components multiply together to
    reconstruct it exactly.

    Parameters
    ----------
    session_catalog : dict
        From discover_smi_sessions.
    min_cc_threshold, cohen_threshold, min_pattern_corr, peak_distance_threshold : float
    activity_method : str

    Returns
    -------
    calib_df : pandas.DataFrame
    """
    rows = []
    skipped = []
    for label, info in session_catalog.items():
        try:
            n_trials = get_session_trial_count(info['tseries_dir'])
        except FileNotFoundError:
            skipped.append(label)
            continue

        session_df = load_session_smi_for_comparison(info['save_path'], session_label=label)
        n_cells = len(session_df)
        avg_cc = session_df['avg_cc'].to_numpy()
        cohen_d = session_df['cohen_d'].to_numpy()
        combined_reliable = session_df['combined_reliable'].to_numpy()

        components = compute_reliability_components_for_session(
            info['tseries_dir'], avg_cc, cohen_d,
            min_cc_threshold=min_cc_threshold, cohen_threshold=cohen_threshold,
            min_pattern_corr=min_pattern_corr, peak_distance_threshold=peak_distance_threshold,
            activity_method=activity_method,
        )

        rows.append({
            'session_label': label,
            'session_type': info['session_type'],
            'n_trials': n_trials,
            'n_cells': n_cells,
            'combined_reliable_fraction': float(np.sum(combined_reliable)) / n_cells,
            'reliable_cells_fraction': components['reliable_cells_fraction'],
            'pattern_reliable_fraction': components['pattern_reliable_fraction'],
            'active_fraction': components['active_fraction'],
        })

    if skipped:
        print(f"WARNING: {len(skipped)} session(s) skipped -- no *preproc*.h5 found: {skipped}")

    calib_df = pd.DataFrame(rows)

    print(f"Built component calibration table for {len(calib_df)} sessions:\n")
    print(calib_df[['session_label', 'session_type', 'n_trials',
                     'combined_reliable_fraction', 'reliable_cells_fraction',
                     'pattern_reliable_fraction', 'active_fraction']].to_string(index=False))

    return calib_df


def test_condition_effect_on_reliability(calib_df, fraction_col, weight_col='n_cells',
                                          reference_level='baseline'):
    """
    Weighted least squares: fraction_col ~ n_trials + C(session_type),
    weighted by n_cells (sessions with more cells get more weight, since
    their fraction estimate is less noisy). Tests whether session_type
    predicts fraction_col beyond what n_trials alone explains.

    Parameters
    ----------
    calib_df : pandas.DataFrame
        From build_reliability_component_calibration (or any table with
        fraction_col, n_trials, session_type, weight_col).
    fraction_col : str
        Which fraction column to test.
    weight_col : str
    reference_level : str
        Which session_type is the reference level for the categorical
        dummy coding.

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


def test_all_reliability_components(calib_df, weight_col='n_cells', reference_level='baseline'):
    """
    Runs test_condition_effect_on_reliability for all four fraction
    columns, side by side.

    Parameters
    ----------
    calib_df : pandas.DataFrame
        From build_reliability_component_calibration.
    weight_col : str
    reference_level : str

    Returns
    -------
    results : dict
        {fraction_col: model_result}.
    """
    fraction_cols = ['combined_reliable_fraction', 'reliable_cells_fraction',
                      'pattern_reliable_fraction', 'active_fraction']
    results = {}
    for col in fraction_cols:
        results[col] = test_condition_effect_on_reliability(
            calib_df, col, weight_col=weight_col, reference_level=reference_level
        )
    return results


# =============================================================================
# Function 4.7 -- compare SMI distributions across conditions
#
# The actual hypothesis test. Filters to 'valid' cells (reliable_valid_cells
# -- reliable AND the SMI curve fit succeeded, the exact mask
# run_smi_analysis_session itself uses). Non-parametric (Kruskal-Wallis /
# Mann-Whitney) since SMI is bounded and likely non-normal.
# =============================================================================

def compare_smi_across_conditions(df, group_col='condition', value_col='SMI', filter_col='valid'):
    """
    Kruskal-Wallis omnibus + pairwise Mann-Whitney U (Holm-corrected)
    across whatever conditions/groups are present.

    Parameters
    ----------
    df : pandas.DataFrame
        One group's table (e.g. from build_comparison_table).
    group_col : str
    value_col : str
    filter_col : str
        Boolean column to filter to before comparing (default 'valid' ==
        reliable_valid_cells).

    Returns
    -------
    result : dict or None
        None (with a printed message) if fewer than 2 categories remain
        after filtering. Otherwise:
        {'group_medians': {cat: median}, 'group_n': {cat: n},
         'omnibus_stat', 'omnibus_p',
         'pairwise': DataFrame(cond_a, cond_b, median_diff, U_stat, p_raw, p_holm)}.
    """
    filtered = df[df[filter_col]]
    categories = [c for c in filtered[group_col].unique() if pd.notna(c)]

    if len(categories) < 2:
        print(f"Only {len(categories)} category(ies) present after filtering on '{filter_col}' "
              f"-- nothing to compare ({categories}).")
        return None

    samples = {cat: filtered.loc[filtered[group_col] == cat, value_col].to_numpy()
               for cat in categories}

    group_medians = {cat: float(np.median(vals)) for cat, vals in samples.items()}
    group_n = {cat: len(vals) for cat, vals in samples.items()}

    omnibus_stat, omnibus_p = kruskal(*samples.values())

    pairwise_rows = []
    for cat_a, cat_b in combinations(categories, 2):
        u_stat, p_raw = mannwhitneyu(samples[cat_a], samples[cat_b], alternative='two-sided')
        pairwise_rows.append({
            'cond_a': cat_a, 'cond_b': cat_b,
            'median_diff': group_medians[cat_a] - group_medians[cat_b],
            'U_stat': u_stat, 'p_raw': p_raw,
        })

    pairwise_df = pd.DataFrame(pairwise_rows)
    if len(pairwise_df) > 0:
        _, p_holm, _, _ = multipletests(pairwise_df['p_raw'], method='holm')
        pairwise_df['p_holm'] = p_holm

    print(f"Categories ({group_col}): {categories}")
    print(f"  n per category: {group_n}")
    print(f"  median {value_col} per category: {group_medians}")
    print(f"  Kruskal-Wallis: H={omnibus_stat:.3f}, p={omnibus_p:.4f}")
    print(f"\n  Pairwise (Holm-corrected):")
    print(pairwise_df.to_string(index=False))

    return {
        'group_medians': group_medians,
        'group_n': group_n,
        'omnibus_stat': omnibus_stat,
        'omnibus_p': omnibus_p,
        'pairwise': pairwise_df,
    }


def _order_categories(values):
    """
    Order category values for plotting/display: condition-style values
    ('baseline'/'saline'/'dcz') get that fixed order; 'DayN'-style values
    (e.g. the baseline group's session_label fallback) get sorted
    numerically by day; anything else keeps first-seen order.

    Parameters
    ----------
    values : array-like

    Returns
    -------
    list
    """
    unique_vals = list(pd.unique(values))
    condition_order = [c for c in ['baseline', 'saline', 'dcz'] if c in unique_vals]
    if len(condition_order) == len(unique_vals):
        return condition_order

    day_pattern = re.compile(r'Day(\d+)', re.IGNORECASE)
    if len(unique_vals) > 0 and all(day_pattern.fullmatch(str(v)) for v in unique_vals):
        return sorted(unique_vals, key=lambda v: int(day_pattern.fullmatch(str(v)).group(1)))

    remaining = [v for v in unique_vals if v not in condition_order]
    return condition_order + remaining


def plot_smi_comparison(df, group_col='condition', value_col='SMI', filter_col='valid', title=''):
    """
    Violin + strip plot of SMI by group_col, filtered to valid cells.

    Parameters
    ----------
    df : pandas.DataFrame
    group_col, value_col, filter_col : str
    title : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    filtered = df[df[filter_col]]
    category_order = _order_categories(filtered[group_col])

    color_by_category = {'baseline': 'tab:blue', 'saline': 'tab:orange', 'dcz': 'tab:green'}

    fig, ax = plt.subplots(figsize=(8, 8))
    data_by_category = [filtered.loc[filtered[group_col] == cat, value_col].to_numpy()
                        for cat in category_order]

    parts = ax.violinplot(data_by_category, showmedians=True)
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(color_by_category.get(category_order[i], 'gray'))
        body.set_alpha(0.4)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data_by_category):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                   color=color_by_category.get(category_order[i], 'gray'),
                   s=15, alpha=0.5)

    ax.set_xticks(range(1, len(category_order) + 1))
    ax.set_xticklabels(category_order)
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


def run_all_group_comparisons(all_group_dfs, group_col='condition', value_col='SMI', filter_col='valid',
                               fallback_group_col='session_label'):
    """
    Loops compare_smi_across_conditions + plot_smi_comparison over every
    group. If a group has fewer than 2 distinct values of group_col (e.g.
    the 'baseline' group, where 'condition' is constant), falls back to
    fallback_group_col instead ('session_label' -- Day1..Day5) rather than
    skipping it outright, so you still see whether SMI drifts across the
    baseline days themselves (a direct check on the "maybe it's
    training-day drift, not the drug" alternative explanation -- see the
    module docstring).

    Each returned result dict includes a 'fig' key (the figure just
    plotted) alongside the stats, so Function 4.8 can save it later
    without needing to replot from scratch.

    Parameters
    ----------
    all_group_dfs : dict
        {group_name: df}, from Function 4.4's run_all_comparisons.
    group_col, value_col, filter_col : str
    fallback_group_col : str or None
        Set to None to restore the old skip-only behavior.

    Returns
    -------
    results : dict
        {group_name: compare_smi_across_conditions(...) result, plus a
        'fig' key} -- only for groups where either grouping produced 2+
        categories.
    """
    results = {}
    for group_name, df in all_group_dfs.items():
        print(f"\n{'='*90}\n{group_name}\n{'='*90}")
        used_group_col = group_col
        result = compare_smi_across_conditions(df, group_col=group_col, value_col=value_col,
                                                filter_col=filter_col)

        if result is None and fallback_group_col is not None and fallback_group_col != group_col:
            print(f"Falling back to grouping by '{fallback_group_col}' instead...")
            used_group_col = fallback_group_col
            result = compare_smi_across_conditions(df, group_col=fallback_group_col, value_col=value_col,
                                                    filter_col=filter_col)

        if result is not None:
            fig = plot_smi_comparison(df, group_col=used_group_col, value_col=value_col,
                                      filter_col=filter_col,
                                      title=f"{group_name} (by {used_group_col})")
            result['fig'] = fig
            results[group_name] = result
            plt.show()

    return results


# =============================================================================
# Function 4.8 -- save everything Phase 4 generates
#
# Nothing in this phase was persisted to disk until this function was
# added -- all_group_dfs, Function 4.7's stats, and every figure only ever
# lived in the kernel's memory. This saves all of it per animal, so future
# comparisons (Phase 5+, or just re-inspecting later) don't require
# re-running the interactive picker from scratch.
# =============================================================================

def save_dataframe_csv(df, output_dir, filename):
    """
    Save a DataFrame to {output_dir}/{filename}, creating output_dir if
    needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    df.to_csv(save_path, index=False)
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


def save_comparison_group_outputs(output_dir, group_name, df, stats_result=None):
    """
    Save one comparison group's outputs: the raw per-cell table (CSV),
    and -- if stats_result is provided (a compare_smi_across_conditions
    result, with the 'fig' key run_all_group_comparisons now adds) -- the
    pairwise stats table (CSV), a summary of medians/n/omnibus (JSON), and
    the figure (PNG).

    Parameters
    ----------
    output_dir : str
    group_name : str
    df : pandas.DataFrame
        The group's raw per-cell table (from all_group_dfs).
    stats_result : dict or None
        From run_all_group_comparisons's per-group result (includes 'fig').

    Returns
    -------
    saved_paths : dict
        {'table', 'pairwise', 'summary', 'figure'} -> path (only keys
        that were actually saved).
    """
    saved_paths = {'table': save_dataframe_csv(df, output_dir, f"{group_name}_comparison_table.csv")}

    if stats_result is not None:
        saved_paths['pairwise'] = save_dataframe_csv(
            stats_result['pairwise'], output_dir, f"{group_name}_smi_pairwise_stats.csv"
        )
        summary = {
            'group_medians': stats_result['group_medians'],
            'group_n': stats_result['group_n'],
            'omnibus_stat': stats_result['omnibus_stat'],
            'omnibus_p': stats_result['omnibus_p'],
        }
        saved_paths['summary'] = save_json(summary, output_dir, f"{group_name}_smi_stats_summary.json")

        fig = stats_result.get('fig')
        if fig is not None:
            saved_paths['figure'] = save_figure_png(fig, output_dir, f"{group_name}_smi_comparison.png")

    return saved_paths


def save_all_comparison_outputs(output_dir, all_group_dfs, smi_comparison_results):
    """
    Loop save_comparison_group_outputs over every group in all_group_dfs.

    Parameters
    ----------
    output_dir : str
    all_group_dfs : dict
        {group_name: df}, from Function 4.4's run_all_comparisons.
    smi_comparison_results : dict
        {group_name: result with 'fig'}, from Function 4.7's
        run_all_group_comparisons.

    Returns
    -------
    saved_paths_by_group : dict
        {group_name: save_comparison_group_outputs(...) result}.
    """
    saved_paths_by_group = {}
    for group_name, df in all_group_dfs.items():
        stats_result = smi_comparison_results.get(group_name)
        saved_paths_by_group[group_name] = save_comparison_group_outputs(
            output_dir, group_name, df, stats_result=stats_result
        )
    print(f"\nSaved outputs for {len(saved_paths_by_group)} group(s) to {output_dir}")
    return saved_paths_by_group


def save_reliability_calibration_outputs(output_dir, calib_df, coeffs, fig):
    """
    Save Function 4.5's outputs: the calibration table (CSV), fit
    coefficients (JSON), and the scatter plot (PNG).
    """
    return {
        'table': save_dataframe_csv(calib_df, output_dir, "reliability_calibration.csv"),
        'coeffs': save_json({'slope': coeffs[0], 'intercept': coeffs[1]},
                             output_dir, "reliability_calibration_fit.json"),
        'figure': save_figure_png(fig, output_dir, "reliability_calibration_plot.png"),
    }


def save_reliability_component_outputs(output_dir, component_calib_df, component_results):
    """
    Save Function 4.6's outputs: the component calibration table (CSV)
    and each fraction column's regression summary (one .txt per column --
    these are printed statsmodels tables, not figures).
    """
    saved_paths = {'table': save_dataframe_csv(component_calib_df, output_dir, "reliability_components.csv")}

    os.makedirs(output_dir, exist_ok=True)
    saved_paths['regressions'] = {}
    for fraction_col, model_result in component_results.items():
        txt_path = os.path.join(output_dir, f"reliability_regression_{fraction_col}.txt")
        with open(txt_path, 'w') as f:
            f.write(str(model_result.summary()))
        print(f"Saved -> {txt_path}")
        saved_paths['regressions'][fraction_col] = txt_path

    return saved_paths


def save_all_phase4_outputs(output_dir, all_group_dfs, smi_comparison_results,
                             calib_df=None, coeffs=None, calib_fig=None,
                             component_calib_df=None, component_results=None):
    """
    Save everything Phase 4 generates for one animal in one call.

    Saved under output_dir:
    - {group}_comparison_table.csv -- the raw per-cell table for every group.
    - {group}_smi_pairwise_stats.csv + {group}_smi_stats_summary.json --
      Function 4.7's pairwise Mann-Whitney table + omnibus/medians/n.
    - {group}_smi_comparison.png -- Function 4.7's violin plot per group.
    - reliability_calibration.csv / _fit.json / _plot.png -- Function
      4.5's outputs, if you ran it.
    - reliability_components.csv + reliability_regression_{fraction_col}.txt
      (one per fraction column) -- Function 4.6's outputs, if you ran it.

    Parameters
    ----------
    output_dir : str
        e.g. os.path.join(ANIMAL_DIR, 'Phase4_SessionComparison_Results').
    all_group_dfs : dict
    smi_comparison_results : dict
        Must carry a 'fig' key per group -- true automatically now that
        Function 4.7's run_all_group_comparisons stores it.
    calib_df, coeffs, calib_fig : optional
        From Function 4.5, if you ran it.
    component_calib_df, component_results : optional
        From Function 4.6, if you ran it.

    Returns
    -------
    saved_paths : dict
    """
    saved_paths = {'comparisons': save_all_comparison_outputs(output_dir, all_group_dfs, smi_comparison_results)}

    if calib_df is not None and coeffs is not None and calib_fig is not None:
        saved_paths['reliability_calibration'] = save_reliability_calibration_outputs(
            output_dir, calib_df, coeffs, calib_fig
        )

    if component_calib_df is not None and component_results is not None:
        saved_paths['reliability_components'] = save_reliability_component_outputs(
            output_dir, component_calib_df, component_results
        )

    print(f"\nAll Phase 4 outputs saved under: {output_dir}")
    return saved_paths


# =============================================================================
# Function 4.9 -- Track A: reload one group's joined per-cell SMI table
# =============================================================================

def load_track_a_smi_table(tracked_groups_dir, group_name):
    """
    Reload one Phase 3 Track A group's joined per-cell SMI table, saved by
    3.SMICalculation.py's save_track_a_joined_table as
    '{group_name}_track_a_smi.csv'. Just a pd.read_csv -- reimplemented
    here rather than imported (digit-prefixed module names aren't cleanly
    importable, same convention used throughout this pipeline).

    Parameters
    ----------
    tracked_groups_dir : str
        The TrackedGroups folder Phase 2/3 saved this group into.
    group_name : str

    Returns
    -------
    joined_df : pandas.DataFrame
    """
    table_path = os.path.join(tracked_groups_dir, f'{group_name}_track_a_smi.csv')
    joined_df = pd.read_csv(table_path)
    print(f"Loaded Track A joined table <- {table_path} ({len(joined_df)} tracked cells)")
    return joined_df


# =============================================================================
# Function 4.10 -- Track A: paired within-cell SMI comparison for one group
#
# Track A's version of Function 4.7's hypothesis test. Same tracked cell's
# SMI compared saline vs. dcz, restricted to cells valid (reliable_valid_
# cells -- reliable AND the SMI curve fit succeeded, the exact filter
# Function 4.7 uses) AND analysis_reliable in BOTH sessions -- see the
# valid_<label> note added to join_smi_to_master_table's docstring.
# =============================================================================

def compare_smi_paired_track_a(joined_df, saline_label, dcz_label):
    """
    Paired Wilcoxon signed-rank (primary -- matches Function 4.7's "SMI is
    bounded/non-normal, use non-parametric" reasoning; unlike Phase 5/6's
    n=5-group paired tests, n here -- tens to ~100 tracked cells -- is
    large enough for Wilcoxon to have real resolving power rather than
    floor out at a fixed minimum p) plus a paired t-test (secondary,
    magnitude-aware -- the same pairing that mattered in Phase 5/6).

    Parameters
    ----------
    joined_df : pandas.DataFrame
        From load_track_a_smi_table.
    saline_label, dcz_label : str
        Must match this group's actual roi_idx_<label>/SMI_<label> column
        suffixes.

    Returns
    -------
    result : dict or None
        None (with a printed message) if fewer than 2 cells are valid &
        analysis_reliable in both sessions. Otherwise: {'n', 'saline_label',
        'dcz_label', 'median_saline', 'median_dcz', 'mean_saline',
        'mean_dcz', 'median_diff', 'mean_diff', 'wilcoxon_stat',
        'wilcoxon_p', 'ttest_stat', 'ttest_p', 'filtered_df'}. median_diff/
        mean_diff are dcz - saline (negative = SMI dropped under DCZ).
    """
    valid_col_saline = f'valid_{saline_label}'
    valid_col_dcz = f'valid_{dcz_label}'
    reliable_col_saline = f'analysis_reliable_{saline_label}'
    reliable_col_dcz = f'analysis_reliable_{dcz_label}'
    smi_col_saline = f'SMI_{saline_label}'
    smi_col_dcz = f'SMI_{dcz_label}'

    required_cols = [valid_col_saline, valid_col_dcz, reliable_col_saline,
                      reliable_col_dcz, smi_col_saline, smi_col_dcz]
    missing = [c for c in required_cols if c not in joined_df.columns]
    if missing:
        raise KeyError(f"joined_df is missing {missing} -- check saline_label/dcz_label match "
                        f"this group's actual session labels, and that Phase 3's "
                        f"join_smi_to_master_table has been re-run since valid_<label> was added.")

    both_ok = (joined_df[valid_col_saline] & joined_df[valid_col_dcz] &
               joined_df[reliable_col_saline] & joined_df[reliable_col_dcz])
    filtered_df = joined_df[both_ok].copy()
    n = len(filtered_df)

    if n < 2:
        print(f"Only {n} cell(s) valid & analysis_reliable in BOTH sessions -- nothing to test.")
        return None

    saline_vals = filtered_df[smi_col_saline].to_numpy()
    dcz_vals = filtered_df[smi_col_dcz].to_numpy()

    wilcoxon_stat, wilcoxon_p = wilcoxon(saline_vals, dcz_vals)
    ttest_stat, ttest_p = ttest_rel(saline_vals, dcz_vals)

    result = {
        'n': n,
        'saline_label': saline_label,
        'dcz_label': dcz_label,
        'median_saline': float(np.median(saline_vals)),
        'median_dcz': float(np.median(dcz_vals)),
        'mean_saline': float(np.mean(saline_vals)),
        'mean_dcz': float(np.mean(dcz_vals)),
        'median_diff': float(np.median(dcz_vals - saline_vals)),
        'mean_diff': float(np.mean(dcz_vals - saline_vals)),
        'wilcoxon_stat': float(wilcoxon_stat),
        'wilcoxon_p': float(wilcoxon_p),
        'ttest_stat': float(ttest_stat),
        'ttest_p': float(ttest_p),
        'filtered_df': filtered_df,
    }

    print(f"Track A paired comparison ({saline_label} vs {dcz_label}): n={n}")
    print(f"  group medians: saline={result['median_saline']:.4f}, dcz={result['median_dcz']:.4f} "
          f"(difference of medians: {result['median_dcz'] - result['median_saline']:.4f} -- "
          f"NOT the same as the line below)")
    print(f"  median of each cell's own (dcz - saline) change: {result['median_diff']:.4f}   "
          f"[paired with Wilcoxon below]")
    print(f"  group means:   saline={result['mean_saline']:.4f}, dcz={result['mean_dcz']:.4f} "
          f"(mean of each cell's own change: {result['mean_diff']:.4f})   [paired with t-test below]")
    print(f"  Wilcoxon signed-rank: stat={wilcoxon_stat:.3f}, p={wilcoxon_p:.4f}")
    print(f"  Paired t-test:        stat={ttest_stat:.3f}, p={ttest_p:.4f}")

    return result


# =============================================================================
# Function 4.11 -- Track A: before/after scatter for one group
# =============================================================================

def plot_smi_paired_track_a(result, title=''):
    """
    SMI_saline (x) vs. SMI_dcz (y) per tracked cell, y=x reference line,
    colored by direction of change. Fits this N (tens to ~100 cells) far
    better than the slope plots Phase 5/6 used for their n=5 comparison
    groups, which would be unreadable at this N.

    Parameters
    ----------
    result : dict
        From compare_smi_paired_track_a (needs 'filtered_df', 'n',
        'saline_label', 'dcz_label').
    title : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    filtered_df = result['filtered_df']
    smi_col_saline = f"SMI_{result['saline_label']}"
    smi_col_dcz = f"SMI_{result['dcz_label']}"

    saline_vals = filtered_df[smi_col_saline].to_numpy()
    dcz_vals = filtered_df[smi_col_dcz].to_numpy()
    decreased = dcz_vals < saline_vals

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(saline_vals[decreased], dcz_vals[decreased], color='tab:red', s=25, alpha=0.6,
               label=f'DCZ < saline (n={int(decreased.sum())})')
    ax.scatter(saline_vals[~decreased], dcz_vals[~decreased], color='tab:blue', s=25, alpha=0.6,
               label=f'DCZ >= saline (n={int((~decreased).sum())})')

    lo = min(saline_vals.min(), dcz_vals.min())
    hi = max(saline_vals.max(), dcz_vals.max())
    ax.plot([lo, hi], [lo, hi], color='gray', linestyle='--', alpha=0.7, label='y = x')

    ax.set_xlabel('SMI (saline)')
    ax.set_ylabel('SMI (dcz)')
    ax.set_title(title or f"Track A paired comparison (n={result['n']})")
    ax.legend(loc='best', fontsize=13)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    return fig


# =============================================================================
# Function 4.12 -- Track A: driver across however many groups you give it
#
# DCZ1/DCZ2/DCZ3 are run separately here, never pooled -- same "don't
# collapse across days" design decision Track B's comparison groups
# already use, so you can see whether the effect replicates group by
# group rather than being averaged away or inflated by one strong day.
# =============================================================================

def run_track_a_paired_comparison(track_a_groups_config):
    """
    Loops Functions 4.9-4.11 over however many Track A groups you give it.

    Parameters
    ----------
    track_a_groups_config : list of dict
        Each: {'group_name', 'tracked_groups_dir', 'saline_label', 'dcz_label'}.

    Returns
    -------
    track_a_results : dict
        {group_name: compare_smi_paired_track_a(...) result, plus a 'fig'
        key} -- only for groups where the comparison actually ran (n >= 2).
    """
    track_a_results = {}
    for cfg in track_a_groups_config:
        print(f"\n{'='*90}\nTRACK A -- {cfg['group_name']}\n{'='*90}")
        joined_df = load_track_a_smi_table(cfg['tracked_groups_dir'], cfg['group_name'])
        result = compare_smi_paired_track_a(joined_df, cfg['saline_label'], cfg['dcz_label'])
        if result is not None:
            result['fig'] = plot_smi_paired_track_a(
                result, title=f"{cfg['group_name']} (Track A, paired within-cell)"
            )
            track_a_results[cfg['group_name']] = result
            plt.show()

    return track_a_results


# =============================================================================
# Function 4.13 -- Track A: save everything above
# =============================================================================

def save_track_a_group_outputs(output_dir, group_name, result):
    """
    Save one Track A group's paired-comparison outputs: the filtered
    per-cell table (CSV), a stats summary (JSON), and the figure (PNG).

    Parameters
    ----------
    output_dir : str
    group_name : str
    result : dict
        From compare_smi_paired_track_a (with 'fig' added by
        run_track_a_paired_comparison).

    Returns
    -------
    saved_paths : dict
    """
    saved_paths = {
        'table': save_dataframe_csv(result['filtered_df'], output_dir,
                                     f"{group_name}_track_a_paired_cells.csv"),
    }

    summary = {k: v for k, v in result.items() if k not in ('filtered_df', 'fig')}
    saved_paths['summary'] = save_json(summary, output_dir, f"{group_name}_track_a_paired_stats.json")

    fig = result.get('fig')
    if fig is not None:
        saved_paths['figure'] = save_figure_png(fig, output_dir, f"{group_name}_track_a_paired_plot.png")

    return saved_paths


def save_all_track_a_outputs(output_dir, track_a_results):
    """
    Loop save_track_a_group_outputs over every group in track_a_results.

    Parameters
    ----------
    output_dir : str
    track_a_results : dict
        {group_name: result with 'fig'}, from run_track_a_paired_comparison.

    Returns
    -------
    saved_paths_by_group : dict
    """
    saved_paths_by_group = {}
    for group_name, result in track_a_results.items():
        saved_paths_by_group[group_name] = save_track_a_group_outputs(output_dir, group_name, result)
    print(f"\nSaved Track A outputs for {len(saved_paths_by_group)} group(s) to {output_dir}")
    return saved_paths_by_group


# =============================================================================
# Driver
# =============================================================================

if __name__ == '__main__':
    # Point this at whichever animal you're processing.
    ANIMAL_DIR = r"D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD"

    # ---- Shared setup (used by both tracks) ----
    session_catalog = discover_smi_sessions(ANIMAL_DIR)

    # =========================================================================
    # TRACK B -- population-level, independent per session
    # =========================================================================
    # Already run + saved for JSY093 (Phase4_SessionComparison_Results/ --
    # this is what Phase 5/6 already consumed) -- skipped here so this run
    # goes straight to TRACK A below without blocking on the interactive
    # group picker. Uncomment for a fresh animal / to redo a group:
    #
    # comparison_groups = collect_comparison_groups_interactively(session_catalog)
    # all_group_dfs = run_all_comparisons(session_catalog, comparison_groups)
    # smi_comparison_results = run_all_group_comparisons(all_group_dfs)
    # OUTPUT_DIR = os.path.join(ANIMAL_DIR, 'Phase4_SessionComparison_Results')
    # saved_paths = save_all_phase4_outputs(OUTPUT_DIR, all_group_dfs, smi_comparison_results)

    # Optional -- Functions 4.5/4.6's reliability-fraction diagnostics (see
    # the module docstring for what these already found on JSY090; mainly
    # useful again if a new animal's saline-vs-DCZ reliable-fraction gap
    # looks unusually large and you want to sanity-check it against trial
    # count before trusting it). Uncomment together with the save call
    # below to also persist these:
    #
    # calib_df, calib_coeffs = build_reliability_trial_count_calibration(session_catalog)
    # calib_fig = plot_reliability_calibration(calib_df, calib_coeffs)
    # plt.show()
    #
    # component_calib_df = build_reliability_component_calibration(session_catalog)
    # component_results = test_all_reliability_components(component_calib_df)
    #
    # saved_paths = save_all_phase4_outputs(
    #     OUTPUT_DIR, all_group_dfs, smi_comparison_results,
    #     calib_df=calib_df, coeffs=calib_coeffs, calib_fig=calib_fig,
    #     component_calib_df=component_calib_df, component_results=component_results,
    # )

    # =========================================================================
    # TRACK A -- tracked-cell, cross-session (paired within-cell SMI
    # comparison on Phase 3's already-joined Track A tables -- no new SMI
    # computation, see compare_smi_paired_track_a). DCZ1/DCZ2/DCZ3 run
    # separately, never pooled, so you can see whether the effect
    # replicates group by group.
    # =========================================================================
    # saline_label/dcz_label per group must match 3.SMICalculation.py's
    # TRACK_A_GROUPS reference_label/session_labels exactly.
    TRACK_A_GROUPS = [
        {
            'group_name': 'DCZ1',
            'saline_label': '260724_JSY_JSY093_LongitudinalImaging_DREADD_Saline_DCZ_1_SALINE',
            'dcz_label': '260724_JSY_JSY093_LongitudinalImaging_DREADD_Saline_DCZ_1_DCZ',
        },
        {
            'group_name': 'DCZ2',
            'saline_label': '260726_JSY_JSY093_LongitudinalImaging_DREADD_Saline_DCZ_2_SALINE',
            'dcz_label': '260726_JSY_JSY093_LongitudinalImaging_DREADD_Saline_DCZ_2_DCZ',
        },
        {
            'group_name': 'DCZ3',
            'saline_label': '260728_JSY_JSY093_LongitudinalImaging_DREADD_Saline_DCZ_3_SALINE',
            'dcz_label': '260728_JSY_JSY093_LongitudinalImaging_DREADD_Saline_DCZ_3_DCZ',
        },
    ]
    for cfg in TRACK_A_GROUPS:
        cfg['tracked_groups_dir'] = os.path.join(
            session_catalog[cfg['saline_label']]['tseries_dir'], 'TrackedGroups'
        )

    track_a_results = run_track_a_paired_comparison(TRACK_A_GROUPS)

    TRACK_A_OUTPUT_DIR = os.path.join(ANIMAL_DIR, 'Phase4_SessionComparison_Results', 'TrackA')
    save_all_track_a_outputs(TRACK_A_OUTPUT_DIR, track_a_results)
