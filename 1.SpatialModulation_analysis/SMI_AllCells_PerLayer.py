"""
SMI_AllCells_PerLayer.py

Companion-file generator: for each existing *_smi_results.h5, writes a
*_smi_results_allcells.h5 that has SMI (and Rp/Rn/preferred position) for
EVERY cell assigned to each layer -- not just the reliability + onset
filtered subset already stored in layer_smi/<layer>/SMI.

Why this is possible without rerunning anything:
  Source files already compute SMI for every iscell==1 cell
  (global_smi/SMI_all_cells, length = n_cells_total). layer_smi/<layer>/SMI
  only keeps the cells that ALSO passed combined_reliable + onset filtering
  (layer_smi/<layer>/reliable_valid_cells). layer_smi/<layer>/cell_indices
  already lists every cell assigned to that layer by depth, so we can index
  straight back into SMI_all_cells to recover SMI for cells that were
  algorithmically valid but failed reliability/onset filtering.

Handling of cells the SMI algorithm itself never computed:
  SMI_Calculation.calculate_SMI_improved zero-initializes SMI/Rp/Rn/
  preferred_position and only overwrites them once it finds a valid
  peak + non-preferred comparison point with Rp+Rn > 0. So a cell that was
  never touched sits at exactly 0.0 for all four -- indistinguishable at
  face value from a genuine near-zero SMI. Because the algorithm's own
  zero_response_sum check guarantees Rp+Rn > 0 for every cell it *did*
  compute, (Rp==0 & Rn==0) alone already perfectly separates "never
  computed" from "computed" with no ambiguity; SMI==0 & preferred_position==0
  are included in the flag too for a belt-and-suspenders check. Flagged
  cells are set to NaN (not left as a misleading 0.0), and `computed_mask`
  records exactly which cells this was applied to.

Does NOT modify or touch the original *_smi_results.h5 files -- everything
is written to a separate companion file alongside the source.

JSY, 2026
"""

import os
import glob
import numpy as np
import h5py


def save_all_cells_smi(smi_results_path, save_path=None, verbose=True):
    """
    Build a companion h5 with SMI (etc.) for every cell in every layer,
    regardless of reliability/onset filtering.

    Parameters
    ----------
    smi_results_path : str
        Path to an existing *_smi_results.h5 file.
    save_path : str, optional
        Output path. Defaults to the same folder, with '_allcells'
        inserted before the extension.

    Returns
    -------
    save_path : str
    """
    if save_path is None:
        base, ext = os.path.splitext(smi_results_path)
        save_path = f"{base}_allcells{ext}"

    with h5py.File(smi_results_path, 'r') as src:
        SMI_all = src['global_smi']['SMI_all_cells'][:]
        Rp_all = src['global_smi']['Rp'][:]
        Rn_all = src['global_smi']['Rn'][:]
        pref_all = src['global_smi']['preferred_positions'][:]
        nonpref_all = src['global_smi']['non_preferred_positions'][:]

        # Cells the SMI algorithm never actually computed (zero-init, untouched)
        never_computed = (SMI_all == 0) & (Rp_all == 0) & (Rn_all == 0) & (pref_all == 0)

        if verbose:
            print(f"\n{os.path.basename(smi_results_path)}")
            print(f"  Total cells: {len(SMI_all)}")
            print(f"  Never computed by algorithm (-> NaN): {np.sum(never_computed)}")

        with h5py.File(save_path, 'w') as dst:
            # Carry over identifying metadata
            for key in ('session_id', 'date', 'animal_id', 'n_cells_total', 'n_cells_analyzed'):
                if key in src.attrs:
                    dst.attrs[key] = src.attrs[key]
            dst.attrs['source_file'] = os.path.basename(smi_results_path)
            dst.attrs['note'] = ('SMI for ALL cells per layer, not just reliability/onset-'
                                  'filtered ones. never_computed heuristic: SMI==0 & Rp==0 & '
                                  'Rn==0 & preferred_position==0 -> NaN.')

            # Copy coordinates / bin_centers / layer_boundaries as-is (self-contained file)
            src.copy('cell_info', dst)
            if 'parameters' in src:
                src.copy('parameters', dst)

            layers_grp = dst.create_group('layer_smi')

            for layer_key in src['layer_smi'].keys():
                src_layer = src['layer_smi'][layer_key]
                cell_indices = src_layer['cell_indices'][:].astype(int)

                smi = SMI_all[cell_indices].copy()
                rp = Rp_all[cell_indices].copy()
                rn = Rn_all[cell_indices].copy()
                pref = pref_all[cell_indices].copy()
                nonpref = nonpref_all[cell_indices].copy()
                computed_mask = ~never_computed[cell_indices]

                smi[~computed_mask] = np.nan
                rp[~computed_mask] = np.nan
                rn[~computed_mask] = np.nan
                pref[~computed_mask] = np.nan
                nonpref[~computed_mask] = np.nan

                # Which of these cells were also in the original reliability-filtered set
                # (reliable_valid_cells stores GLOBAL cell indices, not a boolean mask)
                if 'reliable_valid_cells' in src_layer:
                    reliable_valid_global_idx = src_layer['reliable_valid_cells'][:].astype(int)
                    was_reliable_valid = np.isin(cell_indices, reliable_valid_global_idx)
                else:
                    was_reliable_valid = np.zeros(len(cell_indices), dtype=bool)

                out_layer = layers_grp.create_group(layer_key)
                out_layer.attrs['original_name'] = src_layer.attrs.get('original_name', layer_key)
                out_layer.attrs['n_cells_total'] = len(cell_indices)
                out_layer.attrs['n_cells_computed'] = int(np.sum(computed_mask))
                out_layer.attrs['n_cells_reliable_valid'] = int(np.sum(was_reliable_valid))
                out_layer.attrs['median_smi_all_computed'] = (
                    float(np.nanmedian(smi)) if np.any(computed_mask) else np.nan
                )

                out_layer.create_dataset('cell_indices', data=cell_indices)
                out_layer.create_dataset('SMI', data=smi)
                out_layer.create_dataset('Rp', data=rp)
                out_layer.create_dataset('Rn', data=rn)
                out_layer.create_dataset('preferred_positions', data=pref)
                out_layer.create_dataset('non_preferred_positions', data=nonpref)
                out_layer.create_dataset('computed_mask', data=computed_mask)
                out_layer.create_dataset('was_reliable_valid', data=was_reliable_valid)

                if verbose:
                    print(f"  {layer_key}: {len(cell_indices)} cells total, "
                          f"{np.sum(computed_mask)} computed, "
                          f"{np.sum(was_reliable_valid)} were reliable_valid "
                          f"(median SMI over computed: {out_layer.attrs['median_smi_all_computed']:.3f})")

    if verbose:
        print(f"  Saved: {save_path}")

    return save_path


def batch_save_all_cells_smi(parent_dir, skip_existing=True, verbose=True):
    """Run save_all_cells_smi on every *_smi_results.h5 found under parent_dir."""
    pattern = os.path.join(parent_dir, "**", "*_smi_results.h5")
    all_files = [f for f in glob.glob(pattern, recursive=True) if not f.endswith('_allcells.h5')]

    print(f"Found {len(all_files)} SMI result files under {parent_dir}")

    n_done, n_skipped, n_error = 0, 0, 0
    for path in sorted(all_files):
        base, ext = os.path.splitext(path)
        out_path = f"{base}_allcells{ext}"
        if skip_existing and os.path.exists(out_path):
            n_skipped += 1
            continue
        try:
            save_all_cells_smi(path, save_path=out_path, verbose=verbose)
            n_done += 1
        except Exception as e:
            print(f"  ERROR on {path}: {e}")
            n_error += 1

    print(f"\nBatch complete -- Done: {n_done}  Skipped: {n_skipped}  Errors: {n_error}")


if __name__ == "__main__":
    # Single-session test first (per project convention: verify one before batch)
    test_file = (r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging"
                 r"\251031_JSY_JSY054_SpMod_Day2\TSeries-10312025-1751-001"
                 r"\JSY054_Day2_smi_results.h5")
    save_all_cells_smi(test_file)

    # Once verified, run across everything:
    # batch_save_all_cells_smi(r"D:\V1_SpatialModulation\2p\V1_prism")
