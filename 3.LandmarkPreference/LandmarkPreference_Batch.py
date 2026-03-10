"""
LandmarkPreference_Batch.py
Batch processing script for landmark preference analysis across multiple sessions.
Runs the landmark preference analysis on all specified data files.
Set skip_existing=True (default) to skip sessions that already have results.
JSY, 11/2025
"""

import re
import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import glob
import traceback
from helper import files
from helper import TwoP
from helper.SpatialModulationIndexLayerSpecific import SpatialModulationIndexLayerSpecific as SMI_Layer
from LandmarkPrefernce_SingleSessionAnalysis import run_landmark_analysis


def extract_date_and_session(data_filepath):
    """
    Extract date string and session ID from the data filepath.
    """
    session_folder = os.path.basename(os.path.dirname(data_filepath))
    match = re.match(r'(\d{6})_.*_(Day\d+)', session_folder)
    if match:
        return match.group(1), match.group(2)
    return None, None


def batch_landmark_analysis(skip_existing=True):
    """
    Batch process landmark preference analysis for multiple data files.
    """

    session_dirs = [

        # --- JSY040 ---
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250620_JSY_JSY040_SpatialModulation_Day1_V1Prism\TSeries-06202025-1515-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250622_JSY_JSY040_SpatialModulation_Day3_V1Prism\TSeries-06222025-1550-001',

        # --- JSY041 ---
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250616_JSY_JSY041_SpatialModulation_Day1_V1Prism\TSeries-06162025-1521-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250618_JSY_JSY041_SpatialModulation_Day3_V1Prism\TSeries-06182025-1641-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250620_JSY_JSY041_SpatialModulation_Day5_V1Prism\TSeries-06202025-1515-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250622_JSY_JSY041_SpatialModulation_Day7_V1Prism\TSeries-06222025-1550-001',

        # --- JSY044 ---
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1\TSeries-09062025-1308-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250907_JSY_JSY044_SpaitalModulation_Day2\TSeries-09072025-1257-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3\TSeries-09082025-1540-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250909_JSY_JSY044_SpatialModulation_Day4\TSeries-09092025-1256-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250910_JSY_JSY044_SpatialModulation_Day5\TSeries-09102025-1340-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-001',

        # --- JSY051 ---
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251101_JSY_JSY051_SpMod_Day1\TSeries-11012025-1725-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251102_JSY_JSY051_SpMod_Day2\TSeries-11022025-1642-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251103_JSY_JSY051_SpMod_Day3\TSeries-11032025-1715-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251104_JSY_JSY051_SpMod_Day4\TSeries-11042025-1418-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251105_JSY_JSY051_SpMod_Day5\TSeries-11052025-1512-002',

        # --- JSY052 ---
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251009_JSY_JSY052_SpatialModulation_Day1\TSeries-10092025-1542-002',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251010_JSY_JSY052_SpatialModulation_Day2\TSeries-10102025-0916-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251011_JSY_JSY052_SpatialModulation_Day3\TSeries-10112025-1441-002',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251012_JSY_JSY052_SpatialModulation_Day4\TSeries-10122025-1212-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251013_JSY_JSY052_SpatialModulation_Day5\TSeries-10132025-1236-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251014_JSY_JSY052_SpatialModulation_Day6\TSeries-10142025-1647-003',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251015_JSY_JSY052_SpatialModulation_Day7\TSeries-10152025-1103-001',

        # --- JSY054 ---
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251030_JSY_JSY054_SpMod_Day1\TSeries-10302025-1512-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251031_JSY_JSY054_SpMod_Day2\TSeries-10312025-1751-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251101_JSY_JSY054_SpMod_Day3\TSeries-11012025-1725-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251102_JSY_JSY054_SpMod_Day4\TSeries-11022025-1642-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251103_JSY_JSY054_SpMod_Day5\TSeries-11032025-1715-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251104_JSY_JSY054_SpMod_Day6\TSeries-11042025-1418-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251105_JSY_JSY054_SpMod_Day7\TSeries-11052025-1512-001',

        # --- JSY055 ---
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251205_JSY_JSY055_SpatialModulation_Day1\TSeries-12052025-1740-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251206_JSY_JSY055_SpatialModulation_Day2\TSeries-12062025-1810-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251207_JSY_JSY055_SpatialModulation_Day3\TSeries-12072025-1825-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251208_JSY_JSY055_SpatialModulation_Day4\TSeries-12082025-1633-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251209_JSY_JSY055_SpatialModualtion_Day5\TSeries-12092025-2000-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251210_JSY_JSY055_SpatialModulation_Day6\TSeries-12102025-1702-001',
        r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251211_JSY_JSY055_SpatialModulation_Day7\TSeries-12112025-1631-001',
    ]

    print(f"\n{'='*80}")
    print(f"BATCH LANDMARK PREFERENCE ANALYSIS: {len(session_dirs)} SESSIONS")
    print(f"  skip_existing={skip_existing}")
    print(f"{'='*80}\n")

    n_skipped = 0
    n_done = 0
    n_error = 0
    failed = []

    for idx, data_filepath in enumerate(session_dirs, 1):
        session_folder = os.path.basename(os.path.dirname(data_filepath))
        date_str, session_id = extract_date_and_session(data_filepath)

        print(f"\n{'-'*80}")
        print(f"[{idx}/{len(session_dirs)}] {session_folder}")
        print(f"{'-'*80}")

        if not os.path.isdir(data_filepath):
            print("  Skipped — folder not found")
            n_skipped += 1
            continue

        if skip_existing:
            # Skip only if the figures subfolder already exists (means new-style run is done).
            # Sessions with only an h5 file (old runs, no subfolder) will be re-run
            # to regenerate plots with updated colormaps and folder structure.
            figures_dir = os.path.join(data_filepath, 'LandmarkPreference')
            if os.path.isdir(figures_dir) and glob.glob(os.path.join(figures_dir, '*.png')):
                print("  Skipped — figures folder exists: LandmarkPreference/")
                n_skipped += 1
                continue

        try:
            preproc_files = glob.glob(os.path.join(data_filepath, "*preproc*.h5"))
            if not preproc_files:
                raise ValueError(f"No preprocessed .h5 file found in {data_filepath}")

            preproc_file = preproc_files[0]
            print(f"  Loading: {os.path.basename(preproc_file)}")
            preproc_data = files.read_h5(preproc_file)

            normalized_spatial_activity = preproc_data['norm_spatial_activity']
            bin_centers = preproc_data['bin_centers']
            reliable_valid_cells = preproc_data['combined_reliable']

            twoP_filename = os.path.basename(data_filepath)
            raw_twop_data = TwoP(data_filepath, twoP_filename)
            raw_twop_data.find_files()
            twop_dict = raw_twop_data.calc_dFF()

            numCells = len(twop_dict['stat'])
            im = np.zeros((twop_dict['ops']['Ly'], twop_dict['ops']['Lx']))
            for n in range(numCells):
                ypix = twop_dict['stat'][n]['ypix'][~twop_dict['stat'][n]['overlap']]
                xpix = twop_dict['stat'][n]['xpix'][~twop_dict['stat'][n]['overlap']]
                im[ypix, xpix] = xpix

            med_coords = np.array([cell['med'] for cell in twop_dict['stat']])
            layer_cells, layer_boundaries = SMI_Layer.identify_layers(med_coords)

            landmark_windows_config = [
                {'before': 15, 'after': 10},  # L1 at 25cm: [10, 35]
                {'before': 20, 'after': 10},  # L2 at 55cm: [35, 65]
                {'before': 20, 'after': 10},  # L3 at 85cm: [65, 95]
                {'before': 20, 'after': 10},  # L4 at 115cm: [95, 125]
            ]

            results = run_landmark_analysis(
                normalized_spatial_activity=normalized_spatial_activity,
                bin_centers=bin_centers,
                layer_cells=layer_cells,
                reliable_valid_cells=reliable_valid_cells,
                landmark_positions=[25, 55, 85, 115],
                landmark_windows_config=landmark_windows_config,
                landmark_window=10.0,
                boundary_exclusion=(5, 5),
                exclude_first_bins=5,
                exclude_last_bins=5,
                trials_per_block=20,
                smoothing_sigma=1.0,
                save_path=data_filepath,
                session_id=session_id,
                date_str=date_str
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
    print(f"  Total:     {len(session_dirs)}")
    print(f"  Done:      {n_done}")
    print(f"  Skipped:   {n_skipped}")
    print(f"  Errors:    {n_error}")

    if failed:
        print(f"\nFailed sessions:")
        for item in failed:
            print(f"  {item['session']}: {item['error']}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    batch_landmark_analysis(skip_existing=False)
