"""
SMI_FullSession_Interactive.py
Interactive layer-identification driver for animals where the automatic
identify_layers() pipeline (SMI_FullSession_Batch.py) is unreliable:
  - FOV tilted relative to the cortical layers (needs rotation_deg)
  - Patchy cell density that fools auto-peak detection (needs a manual L4 click)

For each configured session, this:
  1. Loads suite2p outputs directly (no full dFF calc needed) to get med_coords
     and the mean FOV image for the interactive pickers.
  2. Optionally runs estimate_fov_rotation() (click 2 points along a layer
     boundary/landmark) and/or pick_layer4_peak() (click the L4 band on a
     combined density + FOV view).
  3. Runs Run_SMI_Layer_Analysis() with those parameters (default behavior —
     rotation_deg=0, peak_density_method='auto' — is unaffected for every
     other animal; this script is only for the flagged ones).
  4. Reads back the actual L4 boundary used (from the saved smi_results.h5,
     which is correct whether it came from a manual click or from
     auto-detection on the rotated axis) and persists {rotation_deg, peak_y}
     per session to a per-animal JSON store via helper/LayerBoundaryParams.py.
  5. After all of an animal's sessions are done, saves a cross-session
     consistency QC figure — overlaying each session's picked boundary in a
     common (translation-aligned) frame — to visually catch drift or a bad
     click before trusting the layer labels for downstream analysis.

JSY, 2026
"""

import matplotlib
matplotlib.use('Qt5Agg')  # Required for interactive ginput()

import os
import glob
import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

from helper import TwoP
from helper.SpatialModulationIndexLayerSpecific import SpatialModulationIndexLayerSpecific as SMI_Layer
from helper.SMICalculation_LayerSpecific_SingleRecording import Run_SMI_Layer_Analysis
from helper import LayerBoundaryParams as LBP


# ============================================================
# CONFIGURATION
# ============================================================
# Each animal needing manual correction gets one entry. `correct_rotation`
# and `correct_patchy_peak` are picked interactively PER SESSION (tilt/
# density can drift over a longitudinal series) — the flags just say
# whether this animal needs that kind of correction at all.
#
# `um_per_pixel` is optional per animal — omit it to keep the original
# 0.947408849697405 (1.15x objective) default used by every other animal.
# Set it explicitly for FOVs imaged at a different zoom/objective, e.g. the
# V1_prism_DREADD animals (1.08952017715202), so the 70um/150um layer
# thicknesses get converted to pixels correctly.
ANIMALS = [
    {
        'animal_id': 'JSY090',
        'store_dir': r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD',
        'recording_type': 'prism',
        'correct_rotation': False,
        'correct_patchy_peak': True,
        'um_per_pixel': 1.08952017715202,
        'sessions': [

            # r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260719_JSY_JSY090_LongitudinalImaging_DREADD_Day1\TSeries-07192026-0941-001',
            # r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260720_JSY_JSY090_LongitudinalImaging_DREADD_Day2\TSeries-07202026-1009-001',
            # r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260721_JSY_JSY090_LongitudinalImaging_DREADD_Day3\TSeries-07212026-0907-001',
            # r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260722_JSY_JSY090_LongitudinalImaging_DREADD_Day4\TSeries-07222026-0959-001', ** no water delivered session
            r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260722_JSY_JSY090_LongitudinalImaging_DREADD_Day4\TSeries-07222026-1831-002',
            r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260723_JSY_JSY090_LongitudinalImaging_DREADD_Day5\TSeries-07232026-0843-001',
            r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260724_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_1\TSeries-07242026-0809_DCZ-001',
            r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260724_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_1\TSeries-07242026-0809_SALINE-001',
            r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260726_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_2\TSeries-07262026-0755_SAL-001',
            r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY090_V1prism_DREADD\260726_JSY_JSY090_LongitudinalImaging_DREADD_Saline_DCZ_2\TSeries-07262026-0755_DCZ-001',
        ],
    },
    
    # {
    #     'animal_id': 'JSY093',
    #     'store_dir': r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD',
    #     'recording_type': 'prism',
    #     'correct_rotation': True,
    #     'correct_patchy_peak': False,
    #     'sessions': [
    #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260719_JSY_JSY093_LongitudinalImaging_DREADD_Day1\TSeries-07192026-0941-001',       
    #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260720_JSY_JSY093_LongitudinalImaging_DREADD_Day2\TSeries-07202026-1009-001',
    #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260721_JSY_JSY093_LongitudinalImaging_DREADD_Day3\TSeries-07212026-0907-001',
    #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260722_JSY_JSY093_LongitudinalImaging_DREADD_Day4\TSeries-07222026-0959-001',
    #         r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260723_JSY_JSY093_LongitudinalImaging_DREADD_Day5\TSeries-07232026-0843-001',
    #         # r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260724_JSY_JSY093_LongitudinalImaging_DREADD_Saline_DCZ_1\TSeries-07242026-0809_DCZ-001',
    #         # r'D:\V1_SpatialModulation\2p\V1_prism_DREADD\JSY093_V1prism_DREADD\260724_JSY_JSY093_LongitudinalImaging_DREADD_Saline_DCZ_1\TSeries-07242026-0809_SALINE-001'
    #         ],
    #     'um_per_pixel': 1.08952017715202,
    # },
]


def session_label_from_path(session_dir):
    """Day-folder name (one level up from the TSeries folder), e.g. '260724_..._Day1'."""
    return os.path.basename(os.path.dirname(session_dir))


def get_med_coords_and_mean_img(session_dir):
    """Load just what's needed for interactive picking (no dFF/spike calc)."""
    twoP_filename = os.path.basename(session_dir)
    raw = TwoP(session_dir, twoP_filename)
    raw.find_files()
    med_coords = np.array([cell['med'] for cell in raw.stat])
    mean_img = raw.ops['meanImg']
    return med_coords, mean_img


def read_back_peak_y(smi_results_path):
    """
    Recover the L4 peak actually used from a saved smi_results.h5 — correct
    whether it came from auto-detection (on the rotated axis) or a manual
    click, since both end up baked into the saved layer boundaries.
    """
    with h5py.File(smi_results_path, 'r') as f:
        boundaries = f['cell_info']['layer_boundaries'].attrs
        upper = boundaries['L4_upper']
        lower = boundaries['L4_lower']
    return float((upper + lower) / 2)


def process_animal(animal_cfg):
    animal_id = animal_cfg['animal_id']
    store_dir = animal_cfg['store_dir']
    recording_type = animal_cfg.get('recording_type', 'prism')
    correct_rotation = animal_cfg.get('correct_rotation', False)
    correct_patchy_peak = animal_cfg.get('correct_patchy_peak', False)
    um_per_pixel = animal_cfg.get('um_per_pixel', 0.947408849697405)
    sessions = animal_cfg['sessions']

    store_path = os.path.join(store_dir, f"{animal_id}_layer_boundary_params.json")

    print("\n" + "=" * 90)
    print(f" INTERACTIVE LAYER ID: {animal_id}")
    print(f"  correct_rotation={correct_rotation}  correct_patchy_peak={correct_patchy_peak}")
    print("=" * 90)

    session_labels = []
    mean_imgs = []

    for idx, session_dir in enumerate(sessions):
        label = session_label_from_path(session_dir)
        print("\n" + "-" * 80)
        print(f"[{idx+1}/{len(sessions)}] {label}\n  {session_dir}")
        print("-" * 80)

        if not os.path.isdir(session_dir):
            print("  Skipped — folder not found")
            continue

        preproc_files = glob.glob(os.path.join(session_dir, "*preproc*.h5"))
        if not preproc_files:
            print("  Skipped — no preprocessed .h5 found (run Preprocess.py first)")
            continue

        med_coords, mean_img = get_med_coords_and_mean_img(session_dir)
        session_labels.append(label)
        mean_imgs.append(mean_img)

        rotation_deg = 0.0
        if correct_rotation:
            while True:
                print("  Click 2 points along a layer boundary / landmark that should be horizontal...")
                rotation_deg = SMI_Layer.estimate_fov_rotation(mean_img)

                corrected_img = SMI_Layer.rotate_fov_for_preview(mean_img, rotation_deg)
                fig, axes = plt.subplots(1, 2, figsize=(14, 7))
                axes[0].imshow(mean_img, cmap='gray',
                                vmin=np.percentile(mean_img, 2), vmax=np.percentile(mean_img, 98))
                axes[0].set_title('Original FOV')
                axes[0].axis('off')
                axes[1].imshow(corrected_img, cmap='gray',
                                vmin=np.percentile(corrected_img, 2), vmax=np.percentile(corrected_img, 98))
                axes[1].axhline(corrected_img.shape[0] / 2, color='r', linestyle='--', alpha=0.6)
                axes[1].set_title(f'Corrected (rotation_deg={rotation_deg:.2f})\nLayers should now be horizontal')
                axes[1].axis('off')
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.1)

                answer = input("  Does the corrected FOV look right (layers horizontal)? [y/n]: ").strip().lower()
                plt.close(fig)

                if answer.startswith('y'):
                    break
                print("  Redoing the 2-point rotation pick...")

        manual_peak_y = None
        peak_density_method = 'auto'
        if correct_patchy_peak:
            print("  Click the L4 (densest) band on the combined density/FOV view...")
            if rotation_deg:
                theta = np.radians(rotation_deg)
                cy, cx = np.mean(med_coords[:, 0]), np.mean(med_coords[:, 1])
                depth_y = ((med_coords[:, 0] - cy) * np.cos(theta)
                           - (med_coords[:, 1] - cx) * np.sin(theta) + cy)
            else:
                depth_y = med_coords[:, 0]
            density, bins = np.histogram(depth_y, bins=50)
            bin_centers_density = (bins[:-1] + bins[1:]) / 2
            manual_peak_y = SMI_Layer.pick_layer4_peak(med_coords, mean_img, density, bin_centers_density)
            peak_density_method = 'manual_click'

        Run_SMI_Layer_Analysis(
            data_filepath=session_dir,
            recording_type=recording_type,
            rotation_deg=rotation_deg,
            peak_density_method=peak_density_method,
            manual_peak_y=manual_peak_y,
            um_per_pixel=um_per_pixel,
        )

        smi_results = glob.glob(os.path.join(session_dir, "*_smi_results.h5"))
        if smi_results and recording_type == 'prism':
            actual_peak_y = read_back_peak_y(smi_results[0])
            LBP.save_param(store_path, label, rotation_deg, actual_peak_y)

            # Trustworthy final check: real FOV, confirmed layer assignment,
            # boundary lines at the actual angle used (unlike the generic
            # layer_distribution.png, which uses a synthetic footprint map
            # and isn't rotation-aware).
            layer_cells, layer_boundaries = SMI_Layer.identify_layers(
                med_coords, peak_density_method='manual_click',
                rotation_deg=rotation_deg, manual_peak_y=actual_peak_y,
                um_per_pixel=um_per_pixel)
            layers_fig_path = os.path.join(session_dir, 'SMI_Figures', f'{label}_layers_on_fov.png')
            fig = SMI_Layer.plot_layers_on_fov(mean_img, med_coords, layer_cells, layer_boundaries,
                                                rotation_deg=rotation_deg, save_path=layers_fig_path)
            plt.close(fig)

    if len(session_labels) >= 2:
        fig_path = os.path.join(store_dir, f"{animal_id}_boundary_consistency.png")
        params = LBP.load_params(store_path)
        try:
            LBP.plot_boundary_consistency(session_labels, mean_imgs, params, save_path=fig_path)
        except ValueError as e:
            print(f"  Skipped consistency plot: {e}")


if __name__ == '__main__':
    for animal_cfg in ANIMALS:
        process_animal(animal_cfg)
