"""
SMI_Axonal_Batch.py
Batch SMI analysis for axonal imaging sessions (no layer identification).

Calls Run_SMI_AxonalImaging_window_Analysis for each session.
Set skip_existing=True (default) to skip sessions that already have results.

JSY, 03/2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import glob
import traceback

from helper.SMICalculation_LayerSpecific_SingleRecording import Run_SMI_AxonalImaging_window_Analysis


def extract_date_and_session(session_dir):
    parent = os.path.basename(os.path.dirname(session_dir))
    match = re.match(r"(\d{6})_.*_(Day\d+)", parent)
    if match:
        return match.group(1), match.group(2)
    return "Unknown", "Unknown"


def batch_axonal_smi(skip_existing=True, cell_selection='reliability'):

    # BASE = r'D:\V1_SpatialModulation\2p\V1_axonal\JSY060_ChronicImaging_prism'

    # session_dirs = [
    #     # --- JSY061 ---
    #     rf'{BASE}\260225_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day1\TSeries-02252026-0903-001',
    #     rf'{BASE}\260226_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day2\TSeries-02262026-0915-001',  # multi-recording, preproc saved here
    #     rf'{BASE}\260227_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day3\TSeries-02262026-1253-001',
    #     rf'{BASE}\260228_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day4\TSeries-02282026-0919-001',
    #     rf'{BASE}\260301_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day5\TSeries-03012026-0914-002',
    #     rf'{BASE}\260302_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day6\TSeries-03022026-1226-001',
    #     rf'{BASE}\260303_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day7\TSeries-03032026-0817-001',
    # ]
    BASE = r'D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window'

    # Single-recording sessions
    session_dirs = [
        rf'{BASE}\260202_JSY_JSY061_SpMod_AxonalImaging_Day1\TSeries-02022026-1804-001',
        rf'{BASE}\260203_JSY_JSY061_SpMod_AxonalImaging_Day2\TSeries-02032026-1751-001',
        rf'{BASE}\260204_JSY_JSY061_SpMod_AxonalImaging_Day3\TSeries-02042026-2009-001',
        rf'{BASE}\260205_JSY_JSY061_SpMod_AxonalImaging_Day4\TSeries-02052026-1833-002',
        rf'{BASE}\260206_JSY_JSY061_SpMod_AxonalImaging_Day5\TSeries-02062026-1850-001',
        rf'{BASE}\260207_JSY_JSY061_SpMod_AxonalImaging_Day6\TSeries-02072026-2023-001',
        rf'{BASE}\260208_JSY_JSY061_SpMod_AxonalImaging_Day7\TSeries-02082026-1826-001',
    ]

    if cell_selection not in ('reliability', 'skaggs_si'):
        raise ValueError(f"cell_selection must be 'reliability' or 'skaggs_si', got '{cell_selection}'")

    suffix = 'skaggs' if cell_selection == 'skaggs_si' else 'reliability'

    print(f"\n{'='*80}")
    print(f"BATCH AXONAL SMI ANALYSIS: {len(session_dirs)} SESSIONS")
    print(f"  cell_selection={cell_selection}  skip_existing={skip_existing}")
    print(f"{'='*80}\n")

    n_skipped = 0
    n_done = 0
    n_error = 0
    failed = []

    for idx, session_dir in enumerate(session_dirs, 1):
        _, session_id = extract_date_and_session(session_dir)
        session_folder = os.path.basename(os.path.dirname(session_dir))

        print(f"\n{'-'*80}")
        print(f"[{idx}/{len(session_dirs)}] {session_folder}")
        print(f"{'-'*80}")

        if not os.path.isdir(session_dir):
            print("  Skipped — folder not found")
            n_skipped += 1
            continue

        if skip_existing:
            existing = glob.glob(os.path.join(session_dir, f"*_smi_{suffix}_results.h5"))
            if existing:
                print(f"  Skipped — results exist: {os.path.basename(existing[0])}")
                n_skipped += 1
                continue

        try:
            Run_SMI_AxonalImaging_window_Analysis(
                data_filepath=session_dir,
                exclude_first_bins=5,
                exclude_last_bins=5,
                segment_distance=28,
                exclude_start_cm=15,
                exclude_end_cm=10,
                smoothing_sigma=1.0,
                cell_selection=cell_selection,
                save_figures=True,
                verbose=True,
            )
            n_done += 1
            print(f"\n  Successfully processed {session_folder}")

        except Exception as e:
            n_error += 1
            failed.append({'session': session_folder, 'error': str(e)})
            print(f"\n  FAILED: {session_folder}")
            print(f"  Error: {e}")
            traceback.print_exc()

    print(f"\n\n{'='*80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"  Total:    {len(session_dirs)}")
    print(f"  Done:     {n_done}")
    print(f"  Skipped:  {n_skipped}")
    print(f"  Errors:   {n_error}")

    if failed:
        print("\nFailed sessions:")
        for item in failed:
            print(f"  {item['session']}: {item['error']}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # cell_selection: 'reliability' (combined_reliable) or 'skaggs_si' (si_significant_cells)
    batch_axonal_smi(skip_existing=True, cell_selection='skaggs_si')
