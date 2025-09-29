"""
CombineDatasets.py
Script to combine two preprocessed datasets from the same field of view
by matching cells based on their spatial coordinates and morphological features.

This is a simplified version focused on the core functionality needed
for the SimpleCombination.py script.

JSY, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import h5py
from helper import files
import os
from sklearn.metrics import pairwise_distances

class DatasetCombiner:
    
    def __init__(self, dataset1_path, dataset2_path):
        """Initialize the dataset combiner."""
        self.dataset1_path = dataset1_path
        self.dataset2_path = dataset2_path
        
        # Load datasets
        self.dataset1 = self.load_preprocessed_data(dataset1_path)
        self.dataset2 = self.load_preprocessed_data(dataset2_path)
        
        print(f"Dataset 1: {len(self.dataset1['reliable_cells'])} cells")
        print(f"Dataset 2: {len(self.dataset2['reliable_cells'])} cells")
        
    def load_preprocessed_data(self, filepath):
        """Load preprocessed data from .h5 file."""
        with h5py.File(filepath, 'r') as f:
            data = files.recursively_load_dict_contents_from_group(f, '/')
        return data
    
    def extract_cell_features(self, stat_data, ops_data):
        """Extract spatial and morphological features for each cell."""
        n_cells = len(stat_data)
        
        # Initialize feature arrays
        centroids = np.zeros((n_cells, 2))
        areas = np.zeros(n_cells)
        aspect_ratios = np.zeros(n_cells)
        compactness = np.zeros(n_cells)
        
        for i, cell_stat in enumerate(stat_data):
            # Get pixel coordinates
            ypix = cell_stat['ypix']
            xpix = cell_stat['xpix']
            
            # Calculate centroid
            centroids[i, 0] = np.mean(ypix)  # y-coordinate
            centroids[i, 1] = np.mean(xpix)  # x-coordinate
            
            # Calculate area (number of pixels)
            areas[i] = len(ypix)
            
            # Calculate aspect ratio and compactness
            if len(ypix) > 0:
                y_range = np.max(ypix) - np.min(ypix) + 1
                x_range = np.max(xpix) - np.min(xpix) + 1
                aspect_ratios[i] = max(y_range, x_range) / min(y_range, x_range)
                
                # Compactness: area / perimeter^2
                perimeter_approx = 2 * (y_range + x_range)
                compactness[i] = areas[i] / (perimeter_approx ** 2) if perimeter_approx > 0 else 0
        
        # Combine features
        features = np.column_stack([
            centroids,
            areas,
            aspect_ratios,
            compactness
        ])
        
        return features, centroids
    
    def find_matching_cells(self, features1, features2, centroids1, centroids2, 
                          max_distance=5.0, min_similarity=0.7):
        """Find matching cells between two datasets."""
        # Calculate distance matrix between centroids
        distance_matrix = cdist(centroids1, centroids2, metric='euclidean')
        
        # Normalize features for similarity calculation
        def normalize_features(features):
            normalized = features.copy()
            for i in range(features.shape[1]):
                col = features[:, i]
                if np.std(col) > 0:
                    normalized[:, i] = (col - np.mean(col)) / np.std(col)
            return normalized
        
        norm_features1 = normalize_features(features1)
        norm_features2 = normalize_features(features2)
        
        # Calculate feature similarity matrix
        feature_similarity = 1 - pairwise_distances(norm_features1, norm_features2, metric='euclidean')
        feature_similarity = np.clip(feature_similarity, 0, 1)
        
        # Combine distance and feature similarity
        max_dist = np.max(distance_matrix)
        distance_similarity = 1 - (distance_matrix / max_dist)
        
        # Combined similarity score (weighted average)
        combined_similarity = 0.7 * distance_similarity + 0.3 * feature_similarity
        
        # Apply distance constraint
        combined_similarity[distance_matrix > max_distance] = 0
        
        # Apply minimum similarity constraint
        combined_similarity[combined_similarity < min_similarity] = 0
        
        # Use Hungarian algorithm to find optimal matching
        cost_matrix = 1 - combined_similarity
        cost_matrix[combined_similarity == 0] = 1000  # High cost for invalid matches
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Extract valid matches
        matches = []
        match_scores = []
        
        for i, j in zip(row_indices, col_indices):
            if combined_similarity[i, j] > 0:  # Valid match
                matches.append((i, j))
                match_scores.append(combined_similarity[i, j])
        
        return matches, np.array(match_scores)
    
    def visualize_matches(self, centroids1, centroids2, matches, match_scores, 
                         ops1, ops2, save_path=None):
        """Visualize the cell matching results."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Get image dimensions
        ly, lx = ops1['Ly'], ops1['Lx']
        
        # Plot dataset 1 cells
        ax = axes[0]
        ax.scatter(centroids1[:, 1], centroids1[:, 0], c='blue', alpha=0.6, s=20)
        ax.set_xlim(0, lx)
        ax.set_ylim(ly, 0)  # Invert y-axis to match image coordinates
        ax.set_title(f'Dataset 1\n({len(centroids1)} cells)')
        ax.set_xlabel('X coordinate (pixels)')
        ax.set_ylabel('Y coordinate (pixels)')
        
        # Plot dataset 2 cells
        ax = axes[1]
        ax.scatter(centroids2[:, 1], centroids2[:, 0], c='red', alpha=0.6, s=20)
        ax.set_xlim(0, lx)
        ax.set_ylim(ly, 0)
        ax.set_title(f'Dataset 2\n({len(centroids2)} cells)')
        ax.set_xlabel('X coordinate (pixels)')
        ax.set_ylabel('Y coordinate (pixels)')
        
        # Plot matches
        ax = axes[2]
        
        # Plot all cells
        ax.scatter(centroids1[:, 1], centroids1[:, 0], c='lightblue', alpha=0.3, s=15, label='Dataset 1 (unmatched)')
        ax.scatter(centroids2[:, 1], centroids2[:, 0], c='lightcoral', alpha=0.3, s=15, label='Dataset 2 (unmatched)')
        
        # Plot matched cells and connections
        if len(matches) > 0:
            matched_indices1 = [m[0] for m in matches]
            matched_indices2 = [m[1] for m in matches]
            
            # Plot matched cells with different colors
            ax.scatter(centroids1[matched_indices1, 1], centroids1[matched_indices1, 0], 
                      c='blue', s=30, label=f'Dataset 1 matched ({len(matches)})')
            ax.scatter(centroids2[matched_indices2, 1], centroids2[matched_indices2, 0], 
                      c='red', s=30, label=f'Dataset 2 matched ({len(matches)})')
            
            # Draw lines connecting matched cells
            for (i, j), score in zip(matches, match_scores):
                x_coords = [centroids1[i, 1], centroids2[j, 1]]
                y_coords = [centroids1[i, 0], centroids2[j, 0]]
                
                # Color line based on match quality
                color = plt.cm.viridis(score)
                ax.plot(x_coords, y_coords, color=color, alpha=0.7, linewidth=1)
        
        ax.set_xlim(0, lx)
        ax.set_ylim(ly, 0)
        ax.set_title(f'Cell Matches\n({len(matches)} matched pairs)')
        ax.set_xlabel('X coordinate (pixels)')
        ax.set_ylabel('Y coordinate (pixels)')
        ax.legend()
        
        # Add colorbar for match scores
        if len(matches) > 0:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(match_scores), vmax=max(match_scores)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Match Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def combine_spatial_activity(self, spatial_activity1, spatial_activity2, matches):
        """Combine spatial activity data from matched cells."""
        if len(matches) == 0:
            print("No matches found, cannot combine data")
            return None, None
        
        # Get dimensions
        n_trials1 = spatial_activity1.shape[1]
        n_trials2 = spatial_activity2.shape[1]
        n_bins = spatial_activity1.shape[2]
        
        # Verify that spatial bins match
        if spatial_activity1.shape[2] != spatial_activity2.shape[2]:
            print("Warning: Different number of spatial bins in datasets")
            n_bins = min(spatial_activity1.shape[2], spatial_activity2.shape[2])
        
        # Initialize combined activity matrix
        n_matched_cells = len(matches)
        total_trials = n_trials1 + n_trials2
        combined_activity = np.zeros((n_matched_cells, total_trials, n_bins))
        
        # Create cell mapping
        cell_mapping = {
            'combined_to_dataset1': {},
            'combined_to_dataset2': {},
            'dataset1_to_combined': {},
            'dataset2_to_combined': {}
        }
        
        # Combine data for matched cells
        for combined_idx, (idx1, idx2) in enumerate(matches):
            # Combine trials from both datasets
            combined_activity[combined_idx, :n_trials1, :n_bins] = spatial_activity1[idx1, :, :n_bins]
            combined_activity[combined_idx, n_trials1:, :n_bins] = spatial_activity2[idx2, :, :n_bins]
            
            # Update mapping
            cell_mapping['combined_to_dataset1'][combined_idx] = idx1
            cell_mapping['combined_to_dataset2'][combined_idx] = idx2
            cell_mapping['dataset1_to_combined'][idx1] = combined_idx
            cell_mapping['dataset2_to_combined'][idx2] = combined_idx
        
        return combined_activity, cell_mapping
    
    def combine_reliability_data(self, reliable_cells1, reliable_cells2, matches):
        """Combine reliability data for matched cells."""
        combined_reliable = np.zeros(len(matches), dtype=bool)
        
        for combined_idx, (idx1, idx2) in enumerate(matches):
            # Cell is considered reliable if reliable in either dataset
            combined_reliable[combined_idx] = reliable_cells1[idx1] or reliable_cells2[idx2]
        
        return combined_reliable
    
    def combine_datasets(self, max_distance=5.0, min_similarity=0.7, visualize=True, save_visualization=None):
        """Main function to combine the two datasets."""
        print("Extracting cell features from both datasets...")
        
        # Extract features from both datasets
        if 'stat' not in self.dataset1 or 'stat' not in self.dataset2:
            raise ValueError("Cell stat data not found in preprocessed files. Please ensure stat data is saved during preprocessing.")
        
        features1, centroids1 = self.extract_cell_features(self.dataset1['stat'], self.dataset1.get('ops', {'Ly': 512, 'Lx': 512}))
        features2, centroids2 = self.extract_cell_features(self.dataset2['stat'], self.dataset2.get('ops', {'Ly': 512, 'Lx': 512}))
        
        print(f"Dataset 1: {len(features1)} cells")
        print(f"Dataset 2: {len(features2)} cells")
        
        # Find matching cells
        print("Finding matching cells...")
        matches, match_scores = self.find_matching_cells(
            features1, features2, centroids1, centroids2,
            max_distance=max_distance, min_similarity=min_similarity
        )
        
        print(f"Found {len(matches)} matching cells")
        if len(match_scores) > 0:
            print(f"Mean match score: {np.mean(match_scores):.3f}")
            print(f"Match score range: {np.min(match_scores):.3f} - {np.max(match_scores):.3f}")
        
        # Visualize matches if requested
        if visualize:
            ops1 = self.dataset1.get('ops', {'Ly': 512, 'Lx': 512})
            ops2 = self.dataset2.get('ops', {'Ly': 512, 'Lx': 512})
            self.visualize_matches(centroids1, centroids2, matches, match_scores, 
                                 ops1, ops2, save_visualization)
        
        if len(matches) == 0:
            print("No matching cells found. Cannot combine datasets.")
            return None
        
        # Combine spatial activity data
        print("Combining spatial activity data...")
        combined_spatial_activity, cell_mapping = self.combine_spatial_activity(
            self.dataset1['spatial_activity'], 
            self.dataset2['spatial_activity'], 
            matches
        )
        
        # Combine normalized spatial activity if available
        combined_norm_spatial_activity = None
        if 'norm_spatial_activity' in self.dataset1 and 'norm_spatial_activity' in self.dataset2:
            combined_norm_spatial_activity, _ = self.combine_spatial_activity(
                self.dataset1['norm_spatial_activity'], 
                self.dataset2['norm_spatial_activity'], 
                matches
            )
        
        # Combine reliability data
        combined_reliable_cells = self.combine_reliability_data(
            self.dataset1['reliable_cells'],
            self.dataset2['reliable_cells'],
            matches
        )
        
        combined_combined_reliable = None
        if 'combined_reliable' in self.dataset1 and 'combined_reliable' in self.dataset2:
            combined_combined_reliable = self.combine_reliability_data(
                self.dataset1['combined_reliable'],
                self.dataset2['combined_reliable'],
                matches
            )
        
        # Combine other data arrays that depend on cell index
        combined_avg_cc = None
        combined_cohen_d = None
        combined_med_coords = None
        
        if 'avg_cc' in self.dataset1 and 'avg_cc' in self.dataset2:
            combined_avg_cc = np.zeros(len(matches))
            for combined_idx, (idx1, idx2) in enumerate(matches):
                combined_avg_cc[combined_idx] = (self.dataset1['avg_cc'][idx1] + self.dataset2['avg_cc'][idx2]) / 2
        
        if 'cohen_d' in self.dataset1 and 'cohen_d' in self.dataset2:
            combined_cohen_d = np.zeros(len(matches))
            for combined_idx, (idx1, idx2) in enumerate(matches):
                combined_cohen_d[combined_idx] = (self.dataset1['cohen_d'][idx1] + self.dataset2['cohen_d'][idx2]) / 2
        
        if 'med_coords' in self.dataset1 and 'med_coords' in self.dataset2:
            combined_med_coords = np.zeros((len(matches), 2))
            for combined_idx, (idx1, idx2) in enumerate(matches):
                combined_med_coords[combined_idx] = (self.dataset1['med_coords'][idx1] + self.dataset2['med_coords'][idx2]) / 2
        
        # Create combined dataset
        combined_data = {
            'spatial_activity': combined_spatial_activity,
            'norm_spatial_activity': combined_norm_spatial_activity,
            'reliable_cells': combined_reliable_cells,
            'combined_reliable': combined_combined_reliable,
            'avg_cc': combined_avg_cc,
            'cohen_d': combined_cohen_d,
            'bin_centers': self.dataset1['bin_centers'],
            'med_coords': combined_med_coords,
            'cell_mapping': cell_mapping,
            'matches': matches,
            'match_scores': match_scores,
            'n_dataset1_trials': self.dataset1['spatial_activity'].shape[1],
            'n_dataset2_trials': self.dataset2['spatial_activity'].shape[1],
            'dataset1_path': self.dataset1_path,
            'dataset2_path': self.dataset2_path,
            'processing_info': {
                'max_distance': max_distance,
                'min_similarity': min_similarity,
                'n_matched_cells': len(matches),
                'n_dataset1_cells': len(self.dataset1['reliable_cells']),
                'n_dataset2_cells': len(self.dataset2['reliable_cells'])
            }
        }
        
        print(f"\nCombined dataset summary:")
        print(f"  Matched cells: {len(matches)}")
        print(f"  Total trials: {combined_spatial_activity.shape[1]}")
        print(f"  Dataset 1 trials: {self.dataset1['spatial_activity'].shape[1]}")
        print(f"  Dataset 2 trials: {self.dataset2['spatial_activity'].shape[1]}")
        print(f"  Reliable cells: {np.sum(combined_reliable_cells)}")
        
        return combined_data
    
    def save_combined_data(self, combined_data, save_path):
        """Save the combined dataset to an HDF5 file."""
        print(f"Saving combined data to {save_path}")
        files.write_h5(save_path, combined_data)
        print("Combined data saved successfully!")


def combine_datasets_main(dataset1_path, dataset2_path, output_path, 
                         max_distance=5.0, min_similarity=0.7, 
                         visualize=True, save_visualization=None):
    """Main function to combine two datasets from the same field of view."""
    
    # Initialize combiner
    combiner = DatasetCombiner(dataset1_path, dataset2_path)
    
    # Combine datasets
    combined_data = combiner.combine_datasets(
        max_distance=max_distance,
        min_similarity=min_similarity,
        visualize=visualize,
        save_visualization=save_visualization
    )
    
    if combined_data is None:
        print("Failed to combine datasets")
        return None
    
    # Save combined data
    combiner.save_combined_data(combined_data, output_path)
    
    return combined_data