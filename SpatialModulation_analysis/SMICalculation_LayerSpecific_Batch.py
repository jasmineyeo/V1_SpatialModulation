"""
SMICalculation_LayerSpecific_Batch.py
Batch processing script for SMI (layer-specific) analysis using 
SMICalculation_LayerSpecific_SingleRecording.py.

Runs SMI analysis on selected 2p recording sessions.

JSY, 2025
"""

from importlib.metadata import files
import os
import sys
import re
import traceback

# Add repo root for imports
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import glob
from helper import files
# Import main single-session executor
from SMICalculation_LayerSpecific_SingleRecording import Run_SMI_Layer_Analysis


# -----------------------------------------------------------------------------
# Extract date + session ID from folder name
# -----------------------------------------------------------------------------
def extract_date_and_session(session_folder):
    """
    Matches naming style:
    YYMMDD_JSY_JSY052_SpatialModulation_DayX
    """
    match = re.match(r"(\d{6})_.*_(Day\d+)", os.path.basename(session_folder))
    if match:
        return match.group(1), match.group(2)
    return "Unknown", "Unknown"


# -----------------------------------------------------------------------------
# MAIN BATCH FUNCTION
# -----------------------------------------------------------------------------
def batch_SMI_layerspecific_analysis():

    # -------------------------------------------------------------------------
    # DEFINE DATASETS HERE
    # (Folder where .h5 + layer_mapping.json exist)
    # -------------------------------------------------------------------------


    # Define all data file paths to process
    session_dirs = [
        # # JSY054 - Days 1-7
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251030_JSY_JSY054_SpMod_Day1\TSeries-10302025-1512-001",
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251031_JSY_JSY054_SpMod_Day2\TSeries-10312025-1751-001",
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251101_JSY_JSY054_SpMod_Day3\TSeries-11012025-1725-001",
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251102_JSY_JSY054_SpMod_Day4\TSeries-11022025-1642-001",
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251103_JSY_JSY054_SpMod_Day5\TSeries-11032025-1715-001",
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251104_JSY_JSY054_SpMod_Day6\TSeries-11042025-1418-001",
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251105_JSY_JSY054_SpMod_Day7\TSeries-11052025-1512-001",

        # # JSY052 - Days 1-7
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251009_JSY_JSY052_SpatialModulation_Day1\TSeries-10092025-1542-002",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251010_JSY_JSY052_SpatialModulation_Day2\TSeries-10102025-0916-001",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251011_JSY_JSY052_SpatialModulation_Day3\TSeries-10112025-1441-002",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251012_JSY_JSY052_SpatialModulation_Day4\TSeries-10122025-1212-001",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251013_JSY_JSY052_SpatialModulation_Day5\TSeries-10132025-1236-001",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251014_JSY_JSY052_SpatialModulation_Day6\TSeries-10142025-1647-003",
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251015_JSY_JSY052_SpatialModulation_Day7\TSeries-10152025-1103-001',

        # JSY044 - Days 1-7
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1_raw_separateregistration\TSeries-09062025-1308-001",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250907_JSY_JSY044_SpaitalModulation_Day2_raw_separateregistration\TSeries-09072025-1257-001",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3_raw_separateregistration\TSeries-09082025-1540-001",
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250909_JSY_JSY044_SpatialModulation_Day4\TSeries-09092025-1256-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250910_JSY_JSY044_SpatialModulation_Day5\TSeries-09102025-1340-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-001',

        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1_raw_separateregistration\TSeries-09062025-1308-002',
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250907_JSY_JSY044_SpaitalModulation_Day2_raw_separateregistration\TSeries-09072025-1257-002",
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3_raw_separateregistration\TSeries-09082025-1540-002",
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250909_JSY_JSY044_SpatialModulation_Day4\TSeries-09092025-1256-002',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250910_JSY_JSY044_SpatialModulation_Day5\TSeries-09102025-1340-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-002',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-002',
    
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250811_JSY_JSY044_SpatialModulation_Day1\TSeries-08112025-1505-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250813_JSY_JSY044_SpatialModulation_Day3\TSeries-08132025-1456-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250815_JSY_JSY044_SpatialModulation_Day5\TSeries-08152025-1527-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250815_JSY_JSY044_SpatialModulation_Day5\TSeries-08152025-1527-002'
    
        # # JSY051 - Days 1-5
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251101_JSY_JSY051_SpMod_Day1\TSeries-11012025-1725-001",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251102_JSY_JSY051_SpMod_Day2\TSeries-11022025-1642-001",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251103_JSY_JSY051_SpMod_Day3\TSeries-11032025-1715-001",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251104_JSY_JSY051_SpMod_Day4\TSeries-11042025-1418-001",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251105_JSY_JSY051_SpMod_Day5\TSeries-11052025-1512-002",

        # # JSY041 - Days 1, 3, 5, 7
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250616_JSY_JSY041_SpatialModulation_Day1_V1Prism\TSeries-06162025-1521-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250618_JSY_JSY041_SpatialModulation_Day3_V1Prism\TSeries-06182025-1641-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250620_JSY_JSY041_SpatialModulation_Day5_V1Prism\TSeries-06202025-1515-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250622_JSY_JSY041_SpatialModulation_Day7_V1Prism\TSeries-06222025-1550-001',

        # # JSY040 - Days 1, 3
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250620_JSY_JSY040_SpatialModulation_Day1_V1Prism\TSeries-06202025-1515-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250622_JSY_JSY040_SpatialModulation_Day3_V1Prism\TSeries-06222025-1550-001',
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
            print(f"❌ Skipped — folder not found:\n   {session_dir}")
            continue

        try:
            # Check for preprocessed file
            preproc_files = glob.glob(os.path.join(session_dir, "*preproc.h5"))
            if not preproc_files:
                raise ValueError(f"No preprocessed .h5 file found in {session_dir}")

            # Extract metadata for logging only
            date_str, session_id = extract_date_and_session(session_dir)
            print(f"  Date: {date_str}, Session: {session_id}")

            # Run analysis
            print("  🚀 Running SMI layer-specific analysis...")
            Run_SMI_Layer_Analysis(
                data_filepath=session_dir
                )

            print("  ✓ COMPLETE")

        except Exception as e:
            print("\n  ❌ ERROR processing:")
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
