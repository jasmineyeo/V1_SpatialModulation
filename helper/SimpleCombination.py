"""
SimpleCombination.py

Simple script for combining two datasets from the same field of view
and validating the combination quality.

This script ONLY handles:
1. Dataset combination with cell matching
2. Validation of cell matching quality
3. Saving combined dataset

SMI analysis and layer-specific analysis will be done separately.

JSY, 2025
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import h5py

# Add your helper directory to path
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

from helper import files

def combine_datasets_simple(dataset1_path, dataset2_path, output_dir, 
                           animal_id="", max_distance=5.0, min_similarity=0.7):
    """
    Simple function to combine two datasets and validate the result.
    
    Parameters:
    -----------
    dataset1_path : str
        Path to first preprocessed dataset (.h5 file)
    dataset2_path : str
        Path to second preprocessed dataset (.h5 file)
    output_dir : str
        Directory to save outputs
    animal_id : str
        Animal identifier for naming
    max_distance : float
        Maximum distance between cell centroids (pixels)
    min_similarity : float
        Minimum similarity score for matching
        
    Returns:
    --------
    combined_data_path : str
        Path to the saved combined dataset
    """
    
    print("="*60)
    print("DATASET COMBINATION PIPELINE")
    print("="*60)
    print(f"Dataset 1: {os.path.basename(dataset1_path)}")
    print(f"Dataset 2: {os.path.basename(dataset2_path)}")
    print(f"Output Directory: {output_dir}")
    print(f"Parameters: max_distance={max_distance}, min_similarity={min_similarity}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate file paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_data_path = os.path.join(output_dir, f"{animal_id}_combined_dataset.h5")
    matching_viz_path = os.path.join(output_dir, f"{animal_id}_cell_matching.png")
    validation_report_path = os.path.join(output_dir, f"{animal_id}_validation_report.png")
    
    # ==========================================================================
    # STEP 1: COMBINE DATASETS
    # ==========================================================================
    
    print("STEP 1: COMBINING DATASETS")
    print("-" * 30)
    
    try:
        # Import the combination module
        from CombineDatasets import combine_datasets_main
        
        combined_data = combine_datasets_main(
            dataset1_path=dataset1_path,
            dataset2_path=dataset2_path,
            output_path=combined_data_path,
            max_distance=max_distance,
            min_similarity=min_similarity,
            visualize=True,
            save_visualization=matching_viz_path
        )
        
        if combined_data is None:
            print("❌ Dataset combination failed!")
            return None
            
        # Print combination results
        n_matched = combined_data['processing_info']['n_matched_cells']
        n_dataset1 = combined_data['processing_info']['n_dataset1_cells']
        n_dataset2 = combined_data['processing_info']['n_dataset2_cells']
        n_trials = combined_data['spatial_activity'].shape[1]
        n_reliable = np.sum(combined_data['reliable_cells'])
        
        print(f"✅ Successfully combined datasets:")
        print(f"   - Dataset 1: {n_dataset1} cells")
        print(f"   - Dataset 2: {n_dataset2} cells")
        print(f"   - Matched: {n_matched} cells ({n_matched/min(n_dataset1, n_dataset2)*100:.1f}%)")
        print(f"   - Total trials: {n_trials}")
        print(f"   - Reliable cells: {n_reliable}")
        
    except Exception as e:
        print(f"❌ Error in dataset combination: {str(e)}")
        return None
    
    # ==========================================================================
    # STEP 2: VALIDATE COMBINATION
    # ==========================================================================
    
    print("\nSTEP 2: VALIDATING COMBINATION")
    print("-" * 30)
    
    try:
        from CellMatchingValidator import validate_combined_dataset
        
        validation_results = validate_combined_dataset(
            combined_data_path=combined_data_path,
            save_report=True
        )
        
        # Move validation report to desired location
        auto_report_path = combined_data_path.replace('.h5', '_validation_report.png')
        if os.path.exists(auto_report_path) and auto_report_path != validation_report_path:
            import shutil
            shutil.move(auto_report_path, validation_report_path)
        
        # Print validation results
        assessment = validation_results['assessment']
        quality = validation_results['quality_metrics']
        
        print(f"✅ Validation completed:")
        print(f"   - Overall Assessment: {assessment}")
        print(f"   - Spatial Quality: {quality['spatial_quality']:.1f}% (cells with good spatial correlation)")
        print(f"   - Position Quality: {quality['position_quality']:.1f}% (cells with close positions)")
        print(f"   - Activity Correlation: {quality['activity_correlation']:.3f}")
        
        if assessment == "POOR":
            print("\n⚠️  WARNING: Poor matching quality detected!")
            print("   Consider trying different parameters:")
            print(f"   - Increase max_distance (current: {max_distance})")
            print(f"   - Decrease min_similarity (current: {min_similarity})")
            print("   - Check that datasets are from the same field of view")
        elif assessment == "GOOD":
            print("\n✅ Good matching quality - ready for analysis!")
        else:
            print("\n🎉 Excellent matching quality!")
        
    except Exception as e:
        print(f"❌ Error in validation: {str(e)}")
        print("Continuing without validation...")
        validation_results = None
    
    # ==========================================================================
    # STEP 3: SAVE SUMMARY
    # ==========================================================================
    
    print("\nSTEP 3: SAVING SUMMARY")
    print("-" * 30)
    
    # Create summary file
    summary_path = os.path.join(output_dir, f"{animal_id}_combination_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("DATASET COMBINATION SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Animal ID: {animal_id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset 1: {dataset1_path}\n")
        f.write(f"Dataset 2: {dataset2_path}\n\n")
        
        f.write("COMBINATION PARAMETERS:\n")
        f.write(f"  Max distance: {max_distance} pixels\n")
        f.write(f"  Min similarity: {min_similarity}\n\n")
        
        f.write("COMBINATION RESULTS:\n")
        f.write(f"  Dataset 1 cells: {n_dataset1}\n")
        f.write(f"  Dataset 2 cells: {n_dataset2}\n")
        f.write(f"  Matched cells: {n_matched}\n")
        f.write(f"  Match rate: {n_matched/min(n_dataset1, n_dataset2)*100:.1f}%\n")
        f.write(f"  Total trials: {n_trials}\n")
        f.write(f"  Reliable cells: {n_reliable}\n\n")
        
        if validation_results:
            f.write("VALIDATION RESULTS:\n")
            f.write(f"  Overall assessment: {validation_results['assessment']}\n")
            f.write(f"  Spatial quality: {quality['spatial_quality']:.1f}%\n")
            f.write(f"  Position quality: {quality['position_quality']:.1f}%\n")
            f.write(f"  Activity correlation: {quality['activity_correlation']:.3f}\n\n")
        
        f.write("OUTPUT FILES:\n")
        f.write(f"  Combined dataset: {os.path.basename(combined_data_path)}\n")
        f.write(f"  Cell matching plot: {os.path.basename(matching_viz_path)}\n")
        f.write(f"  Validation report: {os.path.basename(validation_report_path)}\n")
        f.write(f"  Summary: {os.path.basename(summary_path)}\n")
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    
    print("\n" + "="*60)
    print("COMBINATION COMPLETED!")
    print("="*60)
    
    print(f"\n📁 Output files saved to: {output_dir}")
    print(f"   📄 Combined dataset: {os.path.basename(combined_data_path)}")
    print(f"   📊 Cell matching plot: {os.path.basename(matching_viz_path)}")
    print(f"   📈 Validation report: {os.path.basename(validation_report_path)}")
    print(f"   📝 Summary: {os.path.basename(summary_path)}")
    
    print(f"\n📈 Results:")
    print(f"   🔗 {n_matched} cells successfully matched")
    print(f"   🧠 {n_reliable} reliable cells available for analysis")
    print(f"   🎯 Match rate: {n_matched/min(n_dataset1, n_dataset2)*100:.1f}%")
    
    if validation_results:
        print(f"   ✅ Quality: {validation_results['assessment']}")
    
    print(f"\n🚀 Ready for SMI analysis!")
    print(f"   Use the combined dataset: {combined_data_path}")
    
    return combined_data_path

def test_combination_parameters(dataset1_path, dataset2_path):
    """
    Test different combination parameters to find optimal settings.
    
    Parameters:
    -----------
    dataset1_path : str
        Path to first dataset
    dataset2_path : str
        Path to second dataset
    """
    
    print("TESTING COMBINATION PARAMETERS")
    print("="*50)
    
    from CombineDatasets import DatasetCombiner
    
    # Parameter combinations to test
    distance_thresholds = [3.0, 5.0, 7.0, 10.0]
    similarity_thresholds = [0.5, 0.6, 0.7, 0.8]
    
    results = []
    best_result = None
    best_score = 0
    
    print("Testing parameter combinations...\n")
    
    for max_dist in distance_thresholds:
        for min_sim in similarity_thresholds:
            try:
                combiner = DatasetCombiner(dataset1_path, dataset2_path)
                
                # Extract features
                features1, centroids1 = combiner.extract_cell_features(
                    combiner.dataset1['stat'], 
                    combiner.dataset1.get('ops', {'Ly': 512, 'Lx': 512})
                )
                features2, centroids2 = combiner.extract_cell_features(
                    combiner.dataset2['stat'], 
                    combiner.dataset2.get('ops', {'Ly': 512, 'Lx': 512})
                )
                
                # Find matches
                matches, match_scores = combiner.find_matching_cells(
                    features1, features2, centroids1, centroids2,
                    max_distance=max_dist, min_similarity=min_sim
                )
                
                n_matches = len(matches)
                mean_score = np.mean(match_scores) if len(match_scores) > 0 else 0
                
                # Calculate a combined score (balance quantity and quality)
                combined_score = n_matches * mean_score
                
                result = {
                    'max_distance': max_dist,
                    'min_similarity': min_sim,
                    'n_matches': n_matches,
                    'mean_score': mean_score,
                    'combined_score': combined_score
                }
                
                results.append(result)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_result = result
                
                print(f"Distance: {max_dist:4.1f}, Similarity: {min_sim:4.2f} → "
                      f"{n_matches:3d} matches, Score: {mean_score:5.3f}")
                
            except Exception as e:
                print(f"Distance: {max_dist:4.1f}, Similarity: {min_sim:4.2f} → Error: {str(e)}")
    
    print("\n" + "="*50)
    print("PARAMETER TESTING RESULTS")
    print("="*50)
    
    if best_result:
        print(f"🏆 BEST PARAMETERS:")
        print(f"   Max distance: {best_result['max_distance']}")
        print(f"   Min similarity: {best_result['min_similarity']}")
        print(f"   Expected matches: {best_result['n_matches']}")
        print(f"   Expected quality: {best_result['mean_score']:.3f}")
        
        return best_result['max_distance'], best_result['min_similarity']
    else:
        print("❌ No valid parameter combinations found!")
        return 5.0, 0.7  # Default values

if __name__ == "__main__":
    # ==========================================================================
    # CONFIGURATION - EDIT THESE PATHS
    # ==========================================================================
    
    # Input datasets (edit these paths)
    dataset1_path = r"F:\2P\spmod\250906_JSY_JSY044_SpatialModulation_Day1\TSeries-09062025-1308-001\09062025_JSY038_preproc.h5"
    dataset2_path = r"F:\2P\spmod\250906_JSY_JSY044_SpatialModulation_Day1\TSeries-09062025-1308-002\09062025_JSY038_preproc.h5"

    # Output directory
    output_dir = r"F:\2P\spmod\250906_JSY_JSY044_SpatialModulation_Day1\Combined_Analysis"
    
    # Animal identifier
    animal_id = "JSY044"
    
    # Combination parameters (or use parameter testing)
    max_distance = 5.0      # Maximum distance between cell centroids (pixels)
    min_similarity = 0.7    # Minimum similarity score for matching
    
    # ==========================================================================
    # OPTION 1: TEST PARAMETERS FIRST (RECOMMENDED)
    # ==========================================================================
    
    print("Do you want to test different parameters first? (y/n): ")
    test_params = input().lower().strip() == 'y'
    
    if test_params:
        print("\nTesting parameters...")
        optimal_distance, optimal_similarity = test_combination_parameters(
            dataset1_path, dataset2_path
        )
        
        print(f"\nUse optimal parameters? Distance: {optimal_distance}, Similarity: {optimal_similarity} (y/n): ")
        use_optimal = input().lower().strip() == 'y'
        
        if use_optimal:
            max_distance = optimal_distance
            min_similarity = optimal_similarity
    
    # ==========================================================================
    # OPTION 2: RUN COMBINATION
    # ==========================================================================
    
    print(f"\nRunning combination with parameters:")
    print(f"  Max distance: {max_distance}")
    print(f"  Min similarity: {min_similarity}")
    print("\nProceed? (y/n): ")
    
    proceed = input().lower().strip() == 'y'
    
    if proceed:
        # Run the combination
        combined_data_path = combine_datasets_simple(
            dataset1_path=dataset1_path,
            dataset2_path=dataset2_path,
            output_dir=output_dir,
            animal_id=animal_id,
            max_distance=max_distance,
            min_similarity=min_similarity
        )
        
        if combined_data_path:
            print(f"\n🎉 SUCCESS! Combined dataset ready at:")
            print(f"   {combined_data_path}")
            print(f"\n📋 Next steps:")
            print(f"   1. Review the validation report")
            print(f"   2. Use the combined dataset for SMI analysis")
            print(f"   3. Run layer-specific analysis")
        else:
            print("\n❌ Combination failed. Check the error messages above.")
    else:
        print("Combination cancelled.")