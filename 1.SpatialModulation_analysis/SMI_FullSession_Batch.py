"""
SMI_FullSession_Batch.py
Batch processing script for SMI (layer-specific) full-session analysis.

Runs SMI analysis on all chronic imaging sessions.
Set skip_existing=True (default) to skip sessions that already have results.

JSY, 2025
"""

from importlib.metadata import files
import os
import re
import traceback
import glob
import sys

sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

from helper import files
from helper.SMICalculation_LayerSpecific_SingleRecording import Run_SMI_Layer_Analysis


def extract_date_and_session(session_dir):
    """Extract date and session ID from parent folder name (one level up from TSeries)."""
    parent = os.path.basename(os.path.dirname(session_dir))
    match = re.match(r"(\d{6})_.*_(Day\d+)", parent)
    if match:
        return match.group(1), match.group(2)
    return "Unknown", "Unknown"


def batch_SMI_layerspecific_analysis(skip_existing=True):

    session_dirs = [

        # # --- JSY040 ---
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250620_JSY_JSY040_SpatialModulation_Day1_V1Prism\TSeries-06202025-1515-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250622_JSY_JSY040_SpatialModulation_Day3_V1Prism\TSeries-06222025-1550-001',

        # # --- JSY041 ---
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250616_JSY_JSY041_SpatialModulation_Day1_V1Prism\TSeries-06162025-1521-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250618_JSY_JSY041_SpatialModulation_Day3_V1Prism\TSeries-06182025-1641-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250620_JSY_JSY041_SpatialModulation_Day5_V1Prism\TSeries-06202025-1515-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250622_JSY_JSY041_SpatialModulation_Day7_V1Prism\TSeries-06222025-1550-001',

        # # --- JSY044 ---
        # r"F:\2P\unprocessed\JSY044\250811_JSY_JSY044_SpatialModulation_Day1\TSeries-08112025-1505-001",
        # r"F:\2P\unprocessed\JSY044\250813_JSY_JSY044_SpatialModulation_Day3\TSeries-08132025-1456-001",
        # r"F:\2P\unprocessed\JSY044\250815_JSY_JSY044_SpatialModulation_Day5\TSeries-08152025-1527-001",
        # r"F:\2P\unprocessed\JSY044\250815_JSY_JSY044_SpatialModulation_Day5\TSeries-08152025-1527-002"
        
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250811_JSY_JSY044_SpatialModulation_Day1\TSeries-08112025-1505-001'
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1\TSeries-09062025-1308-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250907_JSY_JSY044_SpaitalModulation_Day2\TSeries-09072025-1257-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3\TSeries-09082025-1540-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250909_JSY_JSY044_SpatialModulation_Day4\TSeries-09092025-1256-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250910_JSY_JSY044_SpatialModulation_Day5\TSeries-09102025-1340-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-001',

        # # --- JSY051 ---
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251101_JSY_JSY051_SpMod_Day1\TSeries-11012025-1725-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251102_JSY_JSY051_SpMod_Day2\TSeries-11022025-1642-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251103_JSY_JSY051_SpMod_Day3\TSeries-11032025-1715-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251104_JSY_JSY051_SpMod_Day4\TSeries-11042025-1418-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251105_JSY_JSY051_SpMod_Day5\TSeries-11052025-1512-002',

        # # --- JSY052  ---
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251009_JSY_JSY052_SpatialModulation_Day1\TSeries-10092025-1542-002',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251010_JSY_JSY052_SpatialModulation_Day2\TSeries-10102025-0916-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251011_JSY_JSY052_SpatialModulation_Day3\TSeries-10112025-1441-002',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251012_JSY_JSY052_SpatialModulation_Day4\TSeries-10122025-1212-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251013_JSY_JSY052_SpatialModulation_Day5\TSeries-10132025-1236-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251014_JSY_JSY052_SpatialModulation_Day6\TSeries-10142025-1647-003',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251015_JSY_JSY052_SpatialModulation_Day7\TSeries-10152025-1103-001',

        # # --- JSY054 ---
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251030_JSY_JSY054_SpMod_Day1\TSeries-10302025-1512-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251031_JSY_JSY054_SpMod_Day2\TSeries-10312025-1751-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251101_JSY_JSY054_SpMod_Day3\TSeries-11012025-1725-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251102_JSY_JSY054_SpMod_Day4\TSeries-11022025-1642-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251103_JSY_JSY054_SpMod_Day5\TSeries-11032025-1715-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251104_JSY_JSY054_SpMod_Day6\TSeries-11042025-1418-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251105_JSY_JSY054_SpMod_Day7\TSeries-11052025-1512-001',

        # # --- JSY055 (all missing smi_results — priority) ---
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251205_JSY_JSY055_SpatialModulation_Day1\TSeries-12052025-1740-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251206_JSY_JSY055_SpatialModulation_Day2\TSeries-12062025-1810-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251207_JSY_JSY055_SpatialModulation_Day3\TSeries-12072025-1825-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251208_JSY_JSY055_SpatialModulation_Day4\TSeries-12082025-1633-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251209_JSY_JSY055_SpatialModualtion_Day5\TSeries-12092025-2000-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251210_JSY_JSY055_SpatialModulation_Day6\TSeries-12102025-1702-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251211_JSY_JSY055_SpatialModulation_Day7\TSeries-12112025-1631-001',
    ]

    print("\n" + "="*90)
    print(" BATCH: LAYER-SPECIFIC SMI FULL-SESSION ANALYSIS ")
    print("="*90)
    print(f"  skip_existing={skip_existing}")

    n_skipped = 0
    n_done = 0
    n_error = 0

    for idx, session_dir in enumerate(session_dirs):
        print("\n" + "-"*80)
        print(f"[{idx+1}/{len(session_dirs)}] {session_dir}")
        print("-"*80)

        if not os.path.isdir(session_dir):
            print("  Skipped — folder not found")
            n_skipped += 1
            continue

        if skip_existing:
            existing = glob.glob(os.path.join(session_dir, "*_smi_results.h5"))
            if existing:
                print(f"  Skipped — results exist: {os.path.basename(existing[0])}")
                n_skipped += 1
                continue

        try:
            preproc_files = glob.glob(os.path.join(session_dir, "*preproc*.h5"))
            if not preproc_files:
                raise ValueError("No preprocessed .h5 file found")

            date_str, session_id = extract_date_and_session(session_dir)
            print(f"  Date: {date_str}, Session: {session_id}")

            Run_SMI_Layer_Analysis(data_filepath=session_dir)
            print("  COMPLETE")
            n_done += 1

        except Exception as e:
            print(f"\n  ERROR: {e}")
            traceback.print_exc()
            print("  → Continuing to next session...")
            n_error += 1

    print("\n" + "="*90)
    print(f" BATCH COMPLETE — Done: {n_done}  Skipped: {n_skipped}  Errors: {n_error}")
    print("="*90)


if __name__ == "__main__":
    batch_SMI_layerspecific_analysis(skip_existing=False)
