"""
SMI_LapChunk_Batch.py
Batch processing script for within-session lap-chunk SMI analysis using
SMI_LapChunk_SingleRecording.py.

Runs analysis on selected 2p recording sessions.

JSY, 2025
"""

import os
import re
import traceback
import glob
import sys

sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

# Import main single-session executor
from SMI_LapChunk_SingleRecording import run_within_session_analysis


# -----------------------------------------------------------------------------
# Extract date + session ID from folder name
# -----------------------------------------------------------------------------
def extract_date_and_session(session_folder):
    """
    Matches naming style:
    YYMMDD_JSY_JSY0XX_SpatialModulation_DayX
    """
    match = re.match(r"(\d{6})_.*_(Day\d+)", os.path.basename(session_folder))
    if match:
        return match.group(1), match.group(2)
    return "Unknown", "Unknown"


# -----------------------------------------------------------------------------
# MAIN BATCH FUNCTION
# -----------------------------------------------------------------------------
def batch_lapcchunk_analysis(chunk_size=20, min_chunk_size=10,
                              exclude_first_bins=5, exclude_last_bins=5,
                              segment_distance=28, exclude_start_cm=15,
                              exclude_end_cm=10, smoothing_sigma=1.0,
                              save_figures=True):

    # Define all data file paths to process
    session_dirs = [
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1\TSeries-09062025-1308-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250907_JSY_JSY044_SpaitalModulation_Day2\TSeries-09072025-1257-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3\TSeries-09082025-1540-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250909_JSY_JSY044_SpatialModulation_Day4\TSeries-09092025-1256-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250910_JSY_JSY044_SpatialModulation_Day5\TSeries-09102025-1340-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-001',
    ]

    print("\n" + "="*90)
    print(" BATCH: LAP-CHUNK WITHIN-SESSION SMI ANALYSIS ")
    print("="*90)

    for idx, session_dir in enumerate(session_dirs):
        print("\n" + "-"*80)
        print(f"[{idx+1}/{len(session_dirs)}] Processing:")
        print(f"  {session_dir}")
        print("-"*80)

        if not os.path.isdir(session_dir):
            print(f"  Skipped — folder not found:\n   {session_dir}")
            continue

        try:
            # Check for preprocessed file
            preproc_files = glob.glob(os.path.join(session_dir, "*preproc*.h5"))
            if not preproc_files:
                raise ValueError(f"No preprocessed .h5 file found in {session_dir}")

            date_str, session_id = extract_date_and_session(session_dir)
            print(f"  Date: {date_str}, Session: {session_id}")

            run_within_session_analysis(
                data_filepath=session_dir,
                chunk_size=chunk_size,
                min_chunk_size=min_chunk_size,
                exclude_first_bins=exclude_first_bins,
                exclude_last_bins=exclude_last_bins,
                segment_distance=segment_distance,
                exclude_start_cm=exclude_start_cm,
                exclude_end_cm=exclude_end_cm,
                smoothing_sigma=smoothing_sigma,
                save_figures=save_figures,
            )

            print("  COMPLETE")

        except Exception as e:
            print("\n  ERROR processing:")
            print(f"  {session_dir}")
            print("  Error:", e)
            traceback.print_exc()
            print("  -> Continuing to next session...")

    print("\n" + "="*90)
    print(" BATCH COMPLETE — All sessions attempted")
    print("="*90)


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    batch_lapcchunk_analysis()
