"""
CellMatchingValidator.py
Validation script to verify cell matching quality and visualize results.
This helps ensure that the same cells were correctly identified across datasets.

JSY, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import h5py
from helper import files

class CellMatchingValidator:
    
    def __init__(self, combined_data_path):
        """Initialize validator with combined dataset."""
        # Load combined data
        with h5py.File(combined_data_path, 'r') as f:
            self.combined_data = files.recursively_load_dict_contents_from_group(f, '/')
        
        # Load original datasets for comparison
        self.dataset1_path = self.combined_data['dataset1_path']
        self.dataset2_path = self.combined_data['dataset2_path']
        
        with h5py.File(self.dataset1_path, 'r') as f:
            self.dataset1 = files.recursively_load_dict_contents_from_group(f, '/')
        
        with h5py.File(self.dataset2_path, 'r') as f:
            self.dataset2 = files.recursively_load_dict_contents_from_group(f, '/')
    
    def validate_spatial_consistency(self, n_examples=5):
        """Validate that matched cells have consistent spatial tuning patterns."""
        print("VALIDATING SPATIAL CONSISTENCY OF MATCHED CELLS")
        print("="*60)
        
        matches = self.combined_data['matches']
        bin_centers = self.combined_data['bin_centers']
        
        # Calculate correlations between dataset1 and dataset2 for each matched cell
        correlations = []
        
        for combined_idx, (idx1, idx2) in enumerate(matches):
            # Get trial-averaged spatial tuning from each dataset
            tuning1 = np.mean(self.dataset1['norm_spatial_activity'][idx1], axis=0)
            tuning2 = np.mean(self.dataset2['norm_spatial_activity'][idx2], axis=0)
            
            # Calculate correlation
            corr, _ = pearsonr(tuning1, tuning2)
            correlations.append(corr)
        
        correlations = np.array(correlations)
        
        print(f"Spatial tuning correlations between datasets:")
        print(f"  Mean correlation: {np.mean(correlations):.3f}")
        print(f"  Median correlation: {np.median(correlations):.3f}")
        print(f"  Std correlation: {np.std(correlations):.3f}")
        print(f"  Min correlation: {np.min(correlations):.3f}")
        print(f"  Max correlation: {np.max(correlations):.3f}")
        
        # Plot correlation distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(correlations, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(correlations), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(correlations):.3f}')
        plt.xlabel('Spatial Tuning Correlation')
        plt.ylabel('Count')
        plt.title('Distribution of Spatial Tuning Correlations')
        plt.legend()
        
        # Show examples of best and worst matches
        best_indices = np.argsort(correlations)[-n_examples:]
        worst_indices = np.argsort(correlations)[:n_examples]
        
        # Plot best matches
        plt.subplot(2, 2, 2)
        for i, combined_idx in enumerate(best_indices):
            idx1, idx2 = matches[combined_idx]
            tuning1 = np.mean(self.dataset1['norm_spatial_activity'][idx1], axis=0)
            tuning2 = np.mean(self.dataset2['norm_spatial_activity'][idx2], axis=0)
            
            plt.plot(bin_centers, tuning1 + i*0.2, 'b-', alpha=0.7, linewidth=1)
            plt.plot(bin_centers, tuning2 + i*0.2, 'r--', alpha=0.7, linewidth=1)
        
        plt.title(f'Best {n_examples} Matches\n(Blue=Dataset1, Red=Dataset2)')
        plt.xlabel('Position (cm)')
        plt.ylabel('Normalized Activity + Offset')
        
        # Plot worst matches
        plt.subplot(2, 2, 3)
        for i, combined_idx in enumerate(worst_indices):
            idx1, idx2 = matches[combined_idx]
            tuning1 = np.mean(self.dataset1['norm_spatial_activity'][idx1], axis=0)
            tuning2 = np.mean(self.dataset2['norm_spatial_activity'][idx2], axis=0)
            
            plt.plot(bin_centers, tuning1 + i*0.2, 'b-', alpha=0.7, linewidth=1)
            plt.plot(bin_centers, tuning2 + i*0.2, 'r--', alpha=0.7, linewidth=1)
        
        plt.title(f'Worst {n_examples} Matches\n(Blue=Dataset1, Red=Dataset2)')
        plt.xlabel('Position (cm)')
        plt.ylabel('Normalized Activity + Offset')
        
        # Scatter plot of correlations vs match scores
        plt.subplot(2, 2, 4)
        match_scores = self.combined_data['match_scores']
        plt.scatter(match_scores, correlations, alpha=0.6)
        plt.xlabel('Matching Score')
        plt.ylabel('Spatial Tuning Correlation')
        plt.title('Match Score vs Spatial Correlation')
        
        # Add correlation line
        corr_match, p_val = pearsonr(match_scores, correlations)
        plt.text(0.05, 0.95, f'r = {corr_match:.3f}\np = {p_val:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return correlations
    
    def validate_cell_positions(self):
        """Validate that matched cells have similar spatial positions."""
        print("\nVALIDATING CELL POSITION CONSISTENCY")
        print("="*50)
        
        matches = self.combined_data['matches']
        
        # Calculate position differences
        position_diffs = []
        
        for idx1, idx2 in matches:
            # Get cell centroids
            stat1 = self.dataset1['stat'][idx1]
            stat2 = self.dataset2['stat'][idx2]
            
            centroid1 = [np.mean(stat1['ypix']), np.mean(stat1['xpix'])]
            centroid2 = [np.mean(stat2['ypix']), np.mean(stat2['xpix'])]
            
            # Calculate Euclidean distance
            diff = np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
            position_diffs.append(diff)
        
        position_diffs = np.array(position_diffs)
        
        print(f"Position differences (pixels):")
        print(f"  Mean: {np.mean(position_diffs):.2f}")
        print(f"  Median: {np.median(position_diffs):.2f}")
        print(f"  Std: {np.std(position_diffs):.2f}")
        print(f"  Max: {np.max(position_diffs):.2f}")
        
        # Plot position differences
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(position_diffs, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(position_diffs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(position_diffs):.2f}')
        plt.xlabel('Position Difference (pixels)')
        plt.ylabel('Count')
        plt.title('Distribution of Position Differences')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        match_scores = self.combined_data['match_scores']
        plt.scatter(match_scores, position_diffs, alpha=0.6)
        plt.xlabel('Matching Score')
        plt.ylabel('Position Difference (pixels)')
        plt.title('Match Score vs Position Difference')
        
        # Add correlation
        corr, p_val = pearsonr(match_scores, position_diffs)
        plt.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return position_diffs
    
    def validate_activity_levels(self):
        """Validate that matched cells have similar overall activity levels."""
        print("\nVALIDATING ACTIVITY LEVEL CONSISTENCY")
        print("="*50)
        
        matches = self.combined_data['matches']
        
        # Calculate activity levels for each matched cell
        activity1_levels = []
        activity2_levels = []
        
        for idx1, idx2 in matches:
            # Calculate mean activity across all trials and bins
            activity1 = np.mean(self.dataset1['norm_spatial_activity'][idx1])
            activity2 = np.mean(self.dataset2['norm_spatial_activity'][idx2])
            
            activity1_levels.append(activity1)
            activity2_levels.append(activity2)
        
        activity1_levels = np.array(activity1_levels)
        activity2_levels = np.array(activity2_levels)
        
        # Calculate correlation
        corr, p_val = pearsonr(activity1_levels, activity2_levels)
        
        print(f"Activity level correlation: r = {corr:.3f}, p = {p_val:.3f}")
        print(f"Dataset 1 activity - Mean: {np.mean(activity1_levels):.3f}, Std: {np.std(activity1_levels):.3f}")
        print(f"Dataset 2 activity - Mean: {np.mean(activity2_levels):.3f}, Std: {np.std(activity2_levels):.3f}")
        
        # Plot activity comparison
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.scatter(activity1_levels, activity2_levels, alpha=0.6)
        plt.xlabel('Dataset 1 Activity Level')
        plt.ylabel('Dataset 2 Activity Level')
        plt.title(f'Activity Level Correlation\nr = {corr:.3f}')
        
        # Add unity line
        min_val = min(np.min(activity1_levels), np.min(activity2_levels))
        max_val = max(np.max(activity1_levels), np.max(activity2_levels))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        plt.subplot(1, 3, 2)
        plt.hist(activity1_levels, bins=15, alpha=0.7, label='Dataset 1', color='blue')
        plt.hist(activity2_levels, bins=15, alpha=0.7, label='Dataset 2', color='red')
        plt.xlabel('Activity Level')
        plt.ylabel('Count')
        plt.title('Activity Level Distributions')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        activity_ratios = activity2_levels / (activity1_levels + 1e-6)  # Avoid division by zero
        plt.hist(activity_ratios, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(1.0, color='red', linestyle='--', label='Equal activity')
        plt.axvline(np.mean(activity_ratios), color='orange', linestyle='--', 
                   label=f'Mean: {np.mean(activity_ratios):.2f}')
        plt.xlabel('Activity Ratio (Dataset2/Dataset1)')
        plt.ylabel('Count')
        plt.title('Activity Ratio Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return activity1_levels, activity2_levels
    
    def create_validation_report(self, save_path=None):
        """Create a comprehensive validation report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE VALIDATION REPORT")
        print("="*60)
        
        # Run all validations
        spatial_corrs = self.validate_spatial_consistency(n_examples=3)
        position_diffs = self.validate_cell_positions()
        activity1, activity2 = self.validate_activity_levels()
        
        # Summary statistics
        n_matches = len(self.combined_data['matches'])
        n_dataset1_cells = len(self.dataset1['reliable_cells'])
        n_dataset2_cells = len(self.dataset2['reliable_cells'])
        match_rate1 = n_matches / n_dataset1_cells * 100
        match_rate2 = n_matches / n_dataset2_cells * 100
        
        # Create summary figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Summary text
        ax = axes[0, 0]
        ax.axis('off')
        summary_text = f"""
VALIDATION SUMMARY

Datasets:
  Dataset 1: {n_dataset1_cells} cells
  Dataset 2: {n_dataset2_cells} cells
  Matched: {n_matches} cells
  Match Rate: {match_rate1:.1f}% / {match_rate2:.1f}%

Spatial Consistency:
  Mean correlation: {np.mean(spatial_corrs):.3f}
  Median correlation: {np.median(spatial_corrs):.3f}
  Cells with r > 0.5: {np.sum(spatial_corrs > 0.5)} ({np.sum(spatial_corrs > 0.5)/n_matches*100:.1f}%)

Position Consistency:
  Mean distance: {np.mean(position_diffs):.2f} pixels
  Median distance: {np.median(position_diffs):.2f} pixels
  Max distance: {np.max(position_diffs):.2f} pixels

Activity Consistency:
  Activity correlation: {pearsonr(activity1, activity2)[0]:.3f}
  Mean ratio: {np.mean(activity2/activity1):.2f}
        """
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Other plots
        axes[0, 1].hist(spatial_corrs, bins=15, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Spatial Correlations')
        axes[0, 1].set_xlabel('Correlation')
        
        axes[0, 2].hist(position_diffs, bins=15, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Position Differences')
        axes[0, 2].set_xlabel('Distance (pixels)')
        
        axes[1, 0].scatter(activity1, activity2, alpha=0.6)
        axes[1, 0].set_title('Activity Correlation')
        axes[1, 0].set_xlabel('Dataset 1')
        axes[1, 0].set_ylabel('Dataset 2')
        
        match_scores = self.combined_data['match_scores']
        axes[1, 1].scatter(match_scores, spatial_corrs, alpha=0.6)
        axes[1, 1].set_title('Match Score vs Spatial Correlation')
        axes[1, 1].set_xlabel('Match Score')
        axes[1, 1].set_ylabel('Spatial Correlation')
        
        axes[1, 2].scatter(match_scores, position_diffs, alpha=0.6)
        axes[1, 2].set_title('Match Score vs Position Difference')
        axes[1, 2].set_xlabel('Match Score')
        axes[1, 2].set_ylabel('Position Difference (pixels)')
        
        plt.suptitle('Cell Matching Validation Report', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Validation report saved to: {save_path}")
        
        plt.show()
        
        # Print final assessment
        print("\n" + "="*60)
        print("VALIDATION ASSESSMENT")
        print("="*60)
        
        # Quality thresholds
        high_spatial_corr_threshold = 0.5
        low_position_diff_threshold = 3.0
        high_activity_corr_threshold = 0.3
        
        spatial_quality = np.sum(spatial_corrs > high_spatial_corr_threshold) / n_matches * 100
        position_quality = np.sum(position_diffs < low_position_diff_threshold) / n_matches * 100
        activity_corr = pearsonr(activity1, activity2)[0]
        
        print(f"Quality Metrics:")
        print(f"  Spatial consistency: {spatial_quality:.1f}% of cells have r > {high_spatial_corr_threshold}")
        print(f"  Position consistency: {position_quality:.1f}% of cells have distance < {low_position_diff_threshold} pixels")
        print(f"  Activity consistency: Overall correlation r = {activity_corr:.3f}")
        
        # Overall assessment
        if (spatial_quality > 70 and position_quality > 80 and activity_corr > 0.3):
            assessment = "EXCELLENT"
        elif (spatial_quality > 50 and position_quality > 60 and activity_corr > 0.2):
            assessment = "GOOD"
        else:
            assessment = "POOR"
        
        print(f"\nOVERALL ASSESSMENT: {assessment}")
        if assessment == "POOR":
            print("Consider adjusting matching parameters or checking data quality.")
        elif assessment == "GOOD":
            print("Matching quality is acceptable for most analyses.")
        else:
            print("Excellent matching quality! Data is ready for combined analysis.")
        
        return {
            'spatial_correlations': spatial_corrs,
            'position_differences': position_diffs,
            'activity_levels': (activity1, activity2),
            'quality_metrics': {
                'spatial_quality': spatial_quality,
                'position_quality': position_quality,
                'activity_correlation': activity_corr
            },
            'assessment': assessment
        }


def validate_combined_dataset(combined_data_path, save_report=True):
    """
    Convenience function to validate a combined dataset.
    
    Parameters:
    -----------
    combined_data_path : str
        Path to combined dataset file
    save_report : bool
        Whether to save validation report
        
    Returns:
    --------
    validation_results : dict
        Validation results dictionary
    """
    validator = CellMatchingValidator(combined_data_path)
    
    # Generate report save path
    if save_report:
        report_path = combined_data_path.replace('.h5', '_validation_report.png')
    else:
        report_path = None
    
    # Run validation
    validation_results = validator.create_validation_report(save_path=report_path)
    
    return validation_results


def test_matching_robustness(dataset1_path, dataset2_path, n_iterations=10):
    """Test the robustness of cell matching by running multiple iterations."""
    print("TESTING MATCHING ROBUSTNESS")
    print("="*40)
    
    from CombineDatasets import DatasetCombiner
    
    # Parameter variations to test
    distance_variations = [4.0, 5.0, 6.0]
    similarity_variations = [0.65, 0.7, 0.75]
    
    results = []
    
    for max_dist in distance_variations:
        for min_sim in similarity_variations:
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
            
            results.append({
                'max_distance': max_dist,
                'min_similarity': min_sim,
                'n_matches': len(matches),
                'mean_score': np.mean(match_scores) if len(match_scores) > 0 else 0,
                'std_score': np.std(match_scores) if len(match_scores) > 0 else 0
            })
    
    # Analyze stability
    match_counts = [r['n_matches'] for r in results]
    mean_scores = [r['mean_score'] for r in results]
    
    print(f"Match count variation: {np.min(match_counts)} - {np.max(match_counts)} matches")
    print(f"Mean score variation: {np.min(mean_scores):.3f} - {np.max(mean_scores):.3f}")
    print(f"Match count std: {np.std(match_counts):.2f}")
    print(f"Mean score std: {np.std(mean_scores):.3f}")
    
    # Plot robustness results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    params = [f"{r['max_distance']:.1f},{r['min_similarity']:.2f}" for r in results]
    plt.bar(range(len(results)), match_counts)
    plt.xlabel('Parameter Set (max_dist, min_sim)')
    plt.ylabel('Number of Matches')
    plt.title('Matching Robustness: Number of Matches')
    plt.xticks(range(len(results)), params, rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(results)), mean_scores)
    plt.xlabel('Parameter Set (max_dist, min_sim)')
    plt.ylabel('Mean Match Score')
    plt.title('Matching Robustness: Match Quality')
    plt.xticks(range(len(results)), params, rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return results


if __name__ == "__main__":
    # Example usage
    combined_data_path = r"F:\2P\spmod\Combined_Datasets\JSY044_combined_dataset.h5"
    
    # Run validation
    print("Running validation on combined dataset...")
    validation_results = validate_combined_dataset(combined_data_path, save_report=True)