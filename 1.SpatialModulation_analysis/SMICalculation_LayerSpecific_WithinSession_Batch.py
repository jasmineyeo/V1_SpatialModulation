# """
# SMICalculation_LayerSpecific_WithinSession_Batch.py

# Batch processing script for within-session SMI analysis.

# Workflow:
# 1. Run Script 1 (within-session analysis) on all specified recordings
# 2. Run Script 2 (across-days analysis) for each animal

# User defines session_dirs list manually.

# JSY, 2025
# """

# import os
# import sys
# import re
# import traceback
# import glob
# from collections import defaultdict

# # Add repo root for imports
# sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

# # Import analysis scripts
# from SMICalculation_LayerSpecific_WithinSession_SingleRecording import run_within_session_analysis
# from SMICalculation_LayerSpecific_WithinSession_AcrossRecordings import run_across_days_analysis
# from SMICalculation_LayerSpecific_WithinSession_AcrossAnimals import run_across_animals_analysis


# # =============================================================================
# # UTILITY FUNCTIONS
# # =============================================================================

# def extract_animal_and_day(session_dir):
#     """
#     Extract animal ID and day from session directory path.
    
#     Matches patterns like:
#     - JSY052_ChronicImaging/251009_JSY_JSY052_SpatialModulation_Day1
#     - JSY044_ChronicImaging/250906_JSY_JSY044_SpatialModulation_Day1
    
#     Returns:
#         animal_id (str): e.g., 'JSY052'
#         day_str (str): e.g., 'Day1'
#         day_num (int): e.g., 1
#     """
#     # Extract animal ID
#     animal_match = re.search(r'(JSY\d+)', session_dir)
#     animal_id = animal_match.group(1) if animal_match else None
    
#     # Extract day
#     day_match = re.search(r'(Day\d+)', session_dir, re.IGNORECASE)
#     day_str = day_match.group(1) if day_match else None
    
#     # Extract numeric day
#     if day_str:
#         day_num_match = re.search(r'Day(\d+)', day_str, re.IGNORECASE)
#         day_num = int(day_num_match.group(1)) if day_num_match else None
#     else:
#         day_num = None
    
#     return animal_id, day_str, day_num


# def organize_sessions_by_animal(session_dirs):
#     """
#     Organize session directories by animal.
    
#     Parameters:
#         session_dirs (list): List of TSeries directory paths
    
#     Returns:
#         animals_dict (dict): {animal_id: {day_num: session_dir}}
#     """
#     animals_dict = defaultdict(dict)
    
#     for session_dir in session_dirs:
#         animal_id, day_str, day_num = extract_animal_and_day(session_dir)
        
#         if animal_id and day_num is not None:
#             animals_dict[animal_id][day_num] = session_dir
    
#     return dict(animals_dict)


# # =============================================================================
# # BATCH PROCESSING FUNCTIONS
# # =============================================================================

# def batch_script1_within_session(session_dirs, 
#                                  chunk_size=20, 
#                                  min_chunk_size=10,
#                                  exclude_first_bins=5,
#                                  exclude_last_bins=5,
#                                  skip_existing=True):
#     """
#     Batch process Script 1 (within-session analysis) on all recordings.
    
#     Parameters:
#         session_dirs (list): List of TSeries directory paths
#         chunk_size (int): Laps per chunk (default 20)
#         min_chunk_size (int): Minimum laps for last chunk (default 10)
#         exclude_first_bins (int): Onset filter bins (default 5)
#         exclude_last_bins (int): Reward filter bins (default 5)
#         skip_existing (bool): Skip if output already exists (default True)
#     """
#     print("\n" + "="*90)
#     print(" BATCH: SCRIPT 1 - WITHIN-SESSION SMI ANALYSIS ")
#     print("="*90)
#     print(f"Total recordings: {len(session_dirs)}\n")
    
#     results_summary = {'success': [], 'failed': [], 'skipped': []}
    
#     for idx, session_dir in enumerate(session_dirs, 1):
#         print("\n" + "-"*80)
#         print(f"[{idx}/{len(session_dirs)}] Processing:")
#         print(f"  {session_dir}")
#         print("-"*80)
        
#         if not os.path.isdir(session_dir):
#             print(f"  ❌ Skipped — folder not found")
#             results_summary['failed'].append(session_dir)
#             continue
        
#         # Check if already processed
#         if skip_existing:
#             existing_h5 = glob.glob(os.path.join(session_dir, "*_within_session_smi.h5"))
#             if existing_h5:
#                 print(f"  ⏭ Skipped — output already exists: {os.path.basename(existing_h5[0])}")
#                 results_summary['skipped'].append(session_dir)
#                 continue
        
#         # Check for preprocessed file
#         preproc_files = glob.glob(os.path.join(session_dir, "*preproc.h5"))
#         if not preproc_files:
#             print(f"  ❌ Skipped — no preprocessed .h5 file found")
#             results_summary['failed'].append(session_dir)
#             continue
        
#         # Extract metadata
#         animal_id, day_str, day_num = extract_animal_and_day(session_dir)
#         print(f"  Animal: {animal_id}, Day: {day_str}")
        
#         # Run analysis
#         try:
#             print(f"  🚀 Running within-session analysis...")
#             print(f"     Chunk size: {chunk_size} laps, Min: {min_chunk_size} laps")
            
#             result = run_within_session_analysis(
#                 data_filepath=session_dir,
#                 chunk_size=chunk_size,
#                 min_chunk_size=min_chunk_size,
#                 exclude_first_bins=exclude_first_bins,
#                 exclude_last_bins=exclude_last_bins,
#                 save_figures=True
#             )
            
#             if result is not None:
#                 print("  ✓ COMPLETE")
#                 results_summary['success'].append(session_dir)
#             else:
#                 print("  ❌ FAILED (returned None)")
#                 results_summary['failed'].append(session_dir)
        
#         except Exception as e:
#             print("\n  ❌ ERROR:")
#             print(f"  {e}")
#             traceback.print_exc()
#             print("  → Continuing to next session...")
#             results_summary['failed'].append(session_dir)
    
#     # Summary
#     print("\n" + "="*90)
#     print(" SCRIPT 1 BATCH COMPLETE ")
#     print("="*90)
#     print(f"  ✓ Success: {len(results_summary['success'])}")
#     print(f"  ❌ Failed:  {len(results_summary['failed'])}")
#     print(f"  ⏭ Skipped: {len(results_summary['skipped'])}")
#     print(f"  Total:    {len(session_dirs)}")
    
#     if results_summary['failed']:
#         print("\nFailed sessions:")
#         for session in results_summary['failed']:
#             print(f"  - {session}")
    
#     return results_summary


# def batch_script2_across_days(session_dirs, 
#                               focus_days=[1, 2, 3, 4, 5, 6, 7],
#                               skip_existing=True):
#     """
#     Batch process Script 2 (across-days analysis) for each animal.
    
#     Parameters:
#         session_dirs (list): List of TSeries directory paths
#         focus_days (list): Days to analyze (default: 1-7)
#         skip_existing (bool): Skip if output already exists (default True)
#     """
#     print("\n" + "="*90)
#     print(" BATCH: SCRIPT 2 - ACROSS-DAYS ANALYSIS ")
#     print("="*90)
    
#     # Organize by animal
#     animals_dict = organize_sessions_by_animal(session_dirs)
    
#     print(f"Total animals: {len(animals_dict)}")
#     for animal_id in sorted(animals_dict.keys()):
#         days = sorted(animals_dict[animal_id].keys())
#         print(f"  {animal_id}: {len(days)} days (Days {days})")
#     print()
    
#     results_summary = {'success': [], 'failed': [], 'skipped': []}
    
#     for idx, (animal_id, days_dict) in enumerate(sorted(animals_dict.items()), 1):
#         print("\n" + "-"*80)
#         print(f"[{idx}/{len(animals_dict)}] Processing animal: {animal_id}")
#         print("-"*80)
        
#         # Determine animal directory (parent of TSeries folders)
#         first_session = list(days_dict.values())[0]
#         # Navigate up: TSeries -> Day folder -> Animal folder
#         animal_dir = os.path.dirname(os.path.dirname(first_session))
        
#         print(f"  Animal directory: {animal_dir}")
#         print(f"  Days available: {sorted(days_dict.keys())}")
        
#         # Check if already processed
#         if skip_existing:
#             output_h5 = os.path.join(animal_dir, f"{animal_id}_across_days_within_session.h5")
#             if os.path.exists(output_h5):
#                 print(f"  ⏭ Skipped — output already exists")
#                 results_summary['skipped'].append(animal_id)
#                 continue
        
#         # Check if Script 1 outputs exist
#         missing_days = []
#         for day_num, session_dir in days_dict.items():
#             h5_files = glob.glob(os.path.join(session_dir, "*_within_session_smi.h5"))
#             if len(h5_files) == 0:
#                 missing_days.append(day_num)
        
#         if missing_days:
#             print(f"  ⚠ WARNING: Missing Script 1 outputs for days: {missing_days}")
#             if len(missing_days) == len(days_dict):
#                 print(f"  ❌ FAILED — No Script 1 outputs found")
#                 results_summary['failed'].append(animal_id)
#                 continue
#             else:
#                 print(f"  → Proceeding with available days")
        
#         # Run analysis
#         try:
#             print(f"  🚀 Running across-days analysis...")
            
#             result = run_across_days_analysis(
#                 animal_dir=animal_dir,
#                 animal_id=animal_id,
#                 save_path=animal_dir,
#                 focus_days=focus_days
#             )
            
#             if result is not None:
#                 print("  ✓ COMPLETE")
#                 results_summary['success'].append(animal_id)
#             else:
#                 print("  ❌ FAILED (returned None)")
#                 results_summary['failed'].append(animal_id)
        
#         except Exception as e:
#             print("\n  ❌ ERROR:")
#             print(f"  {e}")
#             traceback.print_exc()
#             print("  → Continuing to next animal...")
#             results_summary['failed'].append(animal_id)
    
#     # Summary
#     print("\n" + "="*90)
#     print(" SCRIPT 2 BATCH COMPLETE ")
#     print("="*90)
#     print(f"  ✓ Success: {len(results_summary['success'])}")
#     print(f"  ❌ Failed:  {len(results_summary['failed'])}")
#     print(f"  ⏭ Skipped: {len(results_summary['skipped'])}")
#     print(f"  Total:    {len(animals_dict)}")
    
#     if results_summary['failed']:
#         print("\nFailed animals:")
#         for animal in results_summary['failed']:
#             print(f"  - {animal}")
    
#     return results_summary


# # =============================================================================
# # MAIN BATCH WORKFLOW
# # =============================================================================

# def master_batch_process():
#     """
#     Master batch processing workflow.
    
#     Runs:
#     1. Script 1 on all recordings
#     2. Script 2 for each animal
#     """
    
#     # =========================================================================
#     # DEFINE DATASETS HERE
#     # =========================================================================
    
#     session_dirs = [
#         # JSY054 - Days 1-7
#         r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251030_JSY_JSY054_SpMod_Day1\TSeries-10302025-1512-001",
#         r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251031_JSY_JSY054_SpMod_Day2\TSeries-10312025-1751-001",
#         r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251101_JSY_JSY054_SpMod_Day3\TSeries-11012025-1725-001",
#         # r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251102_JSY_JSY054_SpMod_Day4\TSeries-11022025-1642-001",
#         # r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251103_JSY_JSY054_SpMod_Day5\TSeries-11032025-1715-001",
#         # r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251104_JSY_JSY054_SpMod_Day6\TSeries-11042025-1418-001",
#         # r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251105_JSY_JSY054_SpMod_Day7\TSeries-11052025-1512-001",

#         # JSY052 - Days 1-7
#         r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251009_JSY_JSY052_SpatialModulation_Day1\TSeries-10092025-1542-002",
#         r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251010_JSY_JSY052_SpatialModulation_Day2\TSeries-10102025-0916-001",
#         r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251011_JSY_JSY052_SpatialModulation_Day3\TSeries-10112025-1441-002",
#         # r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251012_JSY_JSY052_SpatialModulation_Day4\TSeries-10122025-1212-001",
#         # r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251013_JSY_JSY052_SpatialModulation_Day5\TSeries-10132025-1236-001",
#         # r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251014_JSY_JSY052_SpatialModulation_Day6\TSeries-10142025-1647-003",
#         # r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251015_JSY_JSY052_SpatialModulation_Day7\TSeries-10152025-1103-001',

#         # JSY044 - Days 1-7 (Set 1)
#         # r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1_raw_separateregistration\TSeries-09062025-1308-001",
#         # r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250907_JSY_JSY044_SpaitalModulation_Day2_raw_separateregistration\TSeries-09072025-1257-001",
#         # r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3_raw_separateregistration\TSeries-09082025-1540-001",
#         # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250909_JSY_JSY044_SpatialModulation_Day4\TSeries-09092025-1256-001',
#         # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250910_JSY_JSY044_SpatialModulation_Day5\TSeries-09102025-1340-001',
#         # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-001',
#         # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-001',

#         # JSY044 - Days 1-7 (Set 2 - alternate recordings)
#         r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1_raw_separateregistration\TSeries-09062025-1308-002',
#         r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250907_JSY_JSY044_SpaitalModulation_Day2_raw_separateregistration\TSeries-09072025-1257-002",
#         r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3_raw_separateregistration\TSeries-09082025-1540-002",
#         # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250909_JSY_JSY044_SpatialModulation_Day4\TSeries-09092025-1256-002',
#         # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250910_JSY_JSY044_SpatialModulation_Day5\TSeries-09102025-1340-001',
#         # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-002',
#         # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-002',
    
#         # JSY044 - Days 1, 3, 5 (Earlier sessions)
#         r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250811_JSY_JSY044_SpatialModulation_Day1\TSeries-08112025-1505-001',
#         r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250813_JSY_JSY044_SpatialModulation_Day3\TSeries-08132025-1456-001',
#         # r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250815_JSY_JSY044_SpatialModulation_Day5\TSeries-08152025-1527-001',
    
#         # JSY051 - Days 1-5
#         r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251101_JSY_JSY051_SpMod_Day1\TSeries-11012025-1725-001",
#         r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251102_JSY_JSY051_SpMod_Day2\TSeries-11022025-1642-001",
#         r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251103_JSY_JSY051_SpMod_Day3\TSeries-11032025-1715-001",
#         # r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251104_JSY_JSY051_SpMod_Day4\TSeries-11042025-1418-001",
#         # r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251105_JSY_JSY051_SpMod_Day5\TSeries-11052025-1512-002",

#         # JSY041 - Days 1, 3, 5, 7
#         r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250616_JSY_JSY041_SpatialModulation_Day1_V1Prism\TSeries-06162025-1521-001',
#         r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250618_JSY_JSY041_SpatialModulation_Day3_V1Prism\TSeries-06182025-1641-001',
#         # r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250620_JSY_JSY041_SpatialModulation_Day5_V1Prism\TSeries-06202025-1515-001',
#         # r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250622_JSY_JSY041_SpatialModulation_Day7_V1Prism\TSeries-06222025-1550-001',

#         # JSY040 - Days 1, 3
#         r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250620_JSY_JSY040_SpatialModulation_Day1_V1Prism\TSeries-06202025-1515-001',
#         r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250622_JSY_JSY040_SpatialModulation_Day3_V1Prism\TSeries-06222025-1550-001',
#     ]
    
#     # =========================================================================
#     # CONFIGURATION
#     # =========================================================================
    
#     # Script 1 parameters
#     CHUNK_SIZE = 20
#     MIN_CHUNK_SIZE = 10
#     EXCLUDE_FIRST_BINS = 5
#     EXCLUDE_LAST_BINS = 5
#     SKIP_EXISTING_SCRIPT1 = True
    
#     # Script 2 parameters
#     FOCUS_DAYS = [1, 2, 3]
#     SKIP_EXISTING_SCRIPT2 = True
    
#     # =========================================================================
#     # RUN BATCH PROCESSING
#     # =========================================================================
    
#     print("\n" + "="*90)
#     print(" MASTER BATCH PROCESSING: WITHIN-SESSION SMI ANALYSIS ")
#     print("="*90)
#     print(f"\nTotal sessions defined: {len(session_dirs)}")
    
#     # Organize by animal for preview
#     animals_preview = organize_sessions_by_animal(session_dirs)
#     print(f"Animals detected: {len(animals_preview)}")
#     for animal_id in sorted(animals_preview.keys()):
#         days = sorted(animals_preview[animal_id].keys())
#         print(f"  {animal_id}: Days {days}")
    
#     print(f"\nConfiguration:")
#     print(f"  Chunk size: {CHUNK_SIZE} laps")
#     print(f"  Min chunk size: {MIN_CHUNK_SIZE} laps")
#     print(f"  Skip existing (Script 1): {SKIP_EXISTING_SCRIPT1}")
#     print(f"  Skip existing (Script 2): {SKIP_EXISTING_SCRIPT2}")
#     print(f"  Focus days: {FOCUS_DAYS}")
    
#     # User confirmation
#     response = input("\nProceed with batch processing? (y/n): ")
#     if response.lower() != 'y':
#         print("Aborted by user.")
#         return
    
#     # Step 1: Run Script 1 on all recordings
#     print("\n" + "="*90)
#     print(" STEP 1: RUNNING SCRIPT 1 ON ALL RECORDINGS ")
#     print("="*90)
    
#     script1_results = batch_script1_within_session(
#         session_dirs,
#         chunk_size=CHUNK_SIZE,
#         min_chunk_size=MIN_CHUNK_SIZE,
#         exclude_first_bins=EXCLUDE_FIRST_BINS,
#         exclude_last_bins=EXCLUDE_LAST_BINS,
#         skip_existing=SKIP_EXISTING_SCRIPT1
#     )
    
#     # Step 2: Run Script 2 for each animal
#     print("\n" + "="*90)
#     print(" STEP 2: RUNNING SCRIPT 2 FOR EACH ANIMAL ")
#     print("="*90)
    
#     script2_results = batch_script2_across_days(
#         session_dirs,
#         focus_days=FOCUS_DAYS,
#         skip_existing=SKIP_EXISTING_SCRIPT2
#     )
    
#     # Final summary
#     print("\n" + "="*90)
#     print(" MASTER BATCH PROCESSING COMPLETE ")
#     print("="*90)
#     print("\nScript 1 (Within-Session):")
#     print(f"  ✓ Success: {len(script1_results['success'])}")
#     print(f"  ❌ Failed:  {len(script1_results['failed'])}")
#     print(f"  ⏭ Skipped: {len(script1_results['skipped'])}")
    
#     print("\nScript 2 (Across-Days):")
#     print(f"  ✓ Success: {len(script2_results['success'])}")
#     print(f"  ❌ Failed:  {len(script2_results['failed'])}")
#     print(f"  ⏭ Skipped: {len(script2_results['skipped'])}")
    
#     print("\n" + "="*90)
#     print(" Next step: Run SMICalculation_WithinSession_AcrossAnimals.py ")
#     print(" to generate population-level analysis ")
#     print("="*90)


# # =============================================================================
# # ENTRY POINT
# # =============================================================================

# if __name__ == "__main__":
#     # Run batch processing (Scripts 1 & 2)
#     master_batch_process()
    
#     # Run Script 3 (across animals)
#     print("\n" + "="*90)
#     print(" STEP 3: RUNNING SCRIPT 3 (ACROSS-ANIMALS ANALYSIS) ")
#     print("="*90)
    
#     # Define parent directory (where all animal folders are located)
#     parent_dir = r"D:\V1_SpatialModulation\2p\V1_prism"
#     save_dir = os.path.join(parent_dir, "across_animals_within_session_analysis")
    
#     try:
#         results = run_across_animals_analysis(
#             parent_dir=parent_dir,
#             save_path=save_dir,
#             focus_days=[1, 2, 3]  # Focus on Days 1-3
#         )
        
#         if results is not None:
#             print("\n✓ Script 3 COMPLETE")
#             print(f"  Outputs saved to: {save_dir}")
#         else:
#             print("\n❌ Script 3 FAILED (returned None)")
    
#     except Exception as e:
#         print("\n❌ Script 3 ERROR:")
#         print(f"  {e}")
#         traceback.print_exc()
    
#     print("\n" + "="*90)
#     print(" ALL PROCESSING COMPLETE ")
#     print("="*90)

"""
Batch_RunRevised_Scripts2and3.py

Run revised Script 2 and Script 3 (skips Script 1 - already done).
"""

import os
import sys

sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

from helper.SMICalculation_LayerSpecific_WithinSession_AcrossRecordings import run_across_days_analysis_revised
from helper.SMICalculation_LayerSpecific_WithinSession_AcrossAnimals import run_across_animals_analysis_revised

# =============================================================================
# RUN REVISED SCRIPT 2 FOR EACH ANIMAL
# =============================================================================

animals = [
    ('JSY054', r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging"),
    ('JSY052', r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging"),
    ('JSY044', r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging"),
    ('JSY051', r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging")
]


# animals = [
#     ('JSY054', r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging"),
#     ('JSY052', r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging"),
#     ('JSY044', r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging"),
#     ('JSY051', r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging"),
#     ('JSY041', r"D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging"),
#     ('JSY040', r"D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging"),
# ]

print("\n" + "="*80)
print(" RUNNING REVISED SCRIPT 2 FOR EACH ANIMAL ")
print("="*80)

for animal_id, animal_dir in animals:
    try:
        run_across_days_analysis_revised(animal_dir, animal_id, focus_days=[1, 2, 3])
    except Exception as e:
        print(f"ERROR with {animal_id}: {e}")

# =============================================================================
# RUN REVISED SCRIPT 3 (ACROSS ANIMALS)
# =============================================================================

print("\n" + "="*80)
print(" RUNNING REVISED SCRIPT 3 (ACROSS ANIMALS) ")
print("="*80)

parent_dir = r"D:\V1_SpatialModulation\2p\V1_prism"
save_dir = os.path.join(parent_dir, "across_animals_within_session_REVISED")

results = run_across_animals_analysis_revised(parent_dir, save_dir, focus_days=[1, 2, 3])

print("\n" + "="*80)
print(" ALL REVISED ANALYSES COMPLETE ")
print("="*80)
