"""
SMICalculation_LayerSpecific_Batch.py
Batch processing script for SMI (layer-specific) analysis using 
SMICalculation_LayerSpecific_SingleRecording.py.

Runs SMI analysis on selected 2p recording sessions.

JSY, 2025
"""

from importlib.metadata import files
import os
import re
import traceback
import glob
import sys

# Add repo root for imports
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

from helper import files

# Import main single-session executor
from helper.SMICalculation_LayerSpecific_SingleRecording import Run_SMI_AxonalImaging_window_Analysis, Run_SMI_Layer_Analysis


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
def batch_SMI_layerspecific_analysis():

    # Define all data file paths to process
    session_dirs = [
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1\TSeries-09062025-1308-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250907_JSY_JSY044_SpaitalModulation_Day2\TSeries-09072025-1257-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3\TSeries-09082025-1540-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250909_JSY_JSY044_SpatialModulation_Day4\TSeries-09092025-1256-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250910_JSY_JSY044_SpatialModulation_Day5\TSeries-09102025-1340-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-001',
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-001"
    ]
    
    print("\n" + "="*90)
    print(" BATCH: LAYER-SPECIFIC SMI ANALYSIS ")
    print("="*90)

    for idx, session_dir in enumerate(session_dirs):
        print("\n" + "-"*80)
        print(f"[{idx+1}/{len(session_dirs)}] Processing:")
        print(f"  {session_dir}")
        print("-"*80)

        if not os.path.isdir(session_dir):
            print(f"Skipped — folder not found:\n   {session_dir}")
            continue

        try:
            # Check for preprocessed file
            preproc_files = glob.glob(os.path.join(session_dir, "*preproc*.h5"))
            if not preproc_files:
                raise ValueError(f"No preprocessed .h5 file found in {session_dir}")

            # Extract metadata for logging only
            date_str, session_id = extract_date_and_session(session_dir)
            print(f"  Date: {date_str}, Session: {session_id}")

            # Run analysis
            print("  Running SMI layer-specific analysis...")
            # Run_SMI_AxonalImaging_window_Analysis(
            #     data_filepath=session_dir
            #     )
            Run_SMI_Layer_Analysis(
                data_filepath=session_dir
                )

            print("COMPLETE")

        except Exception as e:
            print("\n  ERROR processing:")
            print(f"  {session_dir}")
            print("  Error:", e)
            traceback.print_exc()
            print("  → Continuing to next session...")

    print("\n" + "="*90)
    print(" BATCH COMPLETE — All sessions attempted")
    print("="*90)


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    batch_SMI_layerspecific_analysis()
