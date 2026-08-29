"""
SMI_OpenLoopVR_Batch.py
Run layer-specific SMI analysis on open-loop VR sessions.

Wrapper around the existing Run_SMI_Layer_Analysis pipeline.
Handles the open-loop folder naming convention (OpenloopVR_moving /
OpenloopVR_stationary) which does not match the Day\d+ regex used
internally, and renames the output h5 accordingly.

Run AFTER Preprocess_OpenLoopVR.py has generated *_preproc.h5 files.

JSY, 05/2025
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import glob
import traceback

from helper.SMICalculation_LayerSpecific_SingleRecording import Run_SMI_Layer_Analysis


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def parse_openloop_session_info(session_dir):
    """
    Extract animal_id and session_type from an open-loop TSeries path.

    Expected parent folder pattern (one level up from TSeries):
        YYMMDD_JSY_JSYxxx_SpMO_OpenloopVR_moving
        YYMMDD_JSY_JSYxxx_SpMO_OpenloopVR_stationary

    Returns
    -------
    animal_id : str   e.g. 'JSY054'
    session_type : str  'OpenloopVR_moving' | 'OpenloopVR_stationary' | 'OpenloopVR'
    """
    folder = os.path.basename(os.path.dirname(session_dir)).lower()

    animal_match = re.search(r'(jsy\d+)', folder, re.IGNORECASE)
    animal_id    = animal_match.group(1).upper() if animal_match else 'unknown'

    if 'stationary' in folder:
        session_type = 'OpenloopVR_stationary'
    elif 'moving' in folder:
        session_type = 'OpenloopVR_moving'
    else:
        session_type = 'OpenloopVR'

    return animal_id, session_type


def run_openloop_smi(session_dir, skip_existing=True):
    """
    Run SMI layer-specific analysis for one open-loop session.

    Steps:
      1. Check if output already exists (skip_existing).
      2. Call Run_SMI_Layer_Analysis — this computes SMI and saves
         *_smi_results.h5 inside the TSeries folder.
         The file will be named '{animal_id}_unknown_smi_results.h5'
         because the Day\d+ regex does not match open-loop folder names.
      3. Rename the output file to '{animal_id}_{session_type}_smi_results.h5'.

    Parameters
    ----------
    session_dir : str
        Path to the TSeries folder.
    skip_existing : bool
        If True, skip if a *_smi_results.h5 already exists.

    Returns
    -------
    h5_path : str or None
        Path to the renamed h5 file, or None if skipped / failed.
    """
    animal_id, session_type = parse_openloop_session_info(session_dir)
    target_name = f"{animal_id}_{session_type}_smi_results.h5"
    target_path = os.path.join(session_dir, target_name)

    if skip_existing and os.path.isfile(target_path):
        print(f"  Skipped — results exist: {target_name}")
        return target_path

    if not os.path.isdir(session_dir):
        print(f"  Skipped — folder not found: {session_dir}")
        return None

    print(f"  Animal: {animal_id}  |  Session type: {session_type}")

    # Run the existing SMI analysis pipeline
    Run_SMI_Layer_Analysis(
        session_dir,
        exclude_first_bins=5,
        exclude_last_bins=5,
        segment_distance=28,
        exclude_start_cm=15,
        exclude_end_cm=10,
        smoothing_sigma=1.0,
        save_figures=True,
        verbose=True,
    )

    # Find and rename the output file (will be named *_unknown_smi_results.h5)
    produced = glob.glob(os.path.join(session_dir, "*_smi_results.h5"))
    if not produced:
        print(f"  WARNING: no *_smi_results.h5 found after analysis — check for errors above.")
        return None

    # Rename to the correct session-type name
    for f in produced:
        if os.path.basename(f) != target_name:
            os.rename(f, target_path)
            print(f"  Renamed: {os.path.basename(f)} → {target_name}")

    return target_path


# ══════════════════════════════════════════════════════════════════════════════
# Batch
# ══════════════════════════════════════════════════════════════════════════════

def batch_openloop_smi(session_dirs, skip_existing=True):
    print("\n" + "="*90)
    print(" BATCH: OPEN-LOOP VR SMI LAYER-SPECIFIC ANALYSIS ")
    print("="*90)
    print(f"  Sessions: {len(session_dirs)}  |  skip_existing={skip_existing}\n")

    successful, failed, skipped = [], [], []

    for idx, session_dir in enumerate(session_dirs):
        print(f"\n[{idx+1}/{len(session_dirs)}] {os.path.basename(os.path.dirname(session_dir))}")
        print("-"*70)
        try:
            result = run_openloop_smi(session_dir, skip_existing=skip_existing)
            if result is None:
                skipped.append(session_dir)
            else:
                successful.append(session_dir)
        except Exception as e:
            failed.append((session_dir, str(e)))
            print(f"  FAILED: {e}")
            traceback.print_exc()

    print("\n" + "="*90)
    print(f"  Done.  Successful: {len(successful)}  |  Skipped: {len(skipped)}  |  Failed: {len(failed)}")
    for path, err in failed:
        print(f"  FAILED: {os.path.basename(os.path.dirname(path))} — {err}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    session_dirs = [
        # ── JSY044 ────────────────────────────────────────
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\251122_JSY_JSY044_SpMod_OpenLoopVR_Moving\TSeries-11222025-1339-001",
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\251123_JSY_JSY044_SpMod_OpenLoopVR_Stationary\TSeries-11232025-1222-001",

        # ── JSY051 ────────────────────────────────────────
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251106_JSY_JSY051_SpMO_OpenloopVR_moving\TSeries-11062025-1439-002",
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251107_JSY_JSY051_SpMO_OpenloopVR_stationary\TSeries-11072025-1032-001",

        # ── JSY052 ────────────────────────────────────────
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251122_JSY_JSY052_SpMod_OpenLoopVR_Moving\TSeries-11222025-1339-001",
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251123_JSY_JSY052_SpMod_OpenLoopVR_Stationary\TSeries-11232025-1222-002",

        # ── JSY054 ────────────────────────────────────────────────────────
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251106_JSY_JSY054_SpMO_OpenloopVR_moving\TSeries-11062025-1439-001",
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251107_JSY_JSY054_SpMO_OpenloopVR_stationary\TSeries-11072025-1032-001",

        # ── JSY055 ────────────────────────────────────────────────────────
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251212_JSY_JSY055_SpatialModulation_OpenLoopVR_Moving\TSeries-12122025-1421-001",
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251213_JSY_JSY055_SpatialModulation_OpenLoopVR_Stationary\TSeries-12132025-1711-001",

    ]

    batch_openloop_smi(session_dirs, skip_existing=True)
