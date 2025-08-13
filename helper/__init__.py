"""
Preprocessing and analysis for laminar spatial modulation
JSY, 2025
"""


# Import functions
from .files import (
    read_xml,
    write_h5,
    recursively_save_dict_contents_to_group,
    recursively_load_dict_contents_from_group)

from .time import (
    time2float,
    time2str)

from .twop import (
    TwoP)

from .SpikeSmoothing import (
    find_temporal_offset,
    apply_temporal_offset, 
    smooth_spikes, 
    plot_comparison, 
    plot_sample_cells, 
    spatial_smooth)

from .BehavioralDataFiltering import (
    reshape_into_laps, 
    process_data_with_trial_filtering)

from .SpatialDiscretization import (
    spatial_assignment)

from .ReliabilityTesting import (
    test_cell_reliability, 
    test_cell_reliability_with_edge_visualization, 
    plot_edge_activity_distributions, 
    visualize_cell_edge_profiles, 
    normalize_spatial_activity, 
    plot_reliable_cells_side_by_side, 
    plot_reliable_cells_grid, 
    plot_reliable_cells_waterfall, 
    evaluate_pattern_similarity, 
    combined_reliability_test)

from .ResponseVisualization import (
    create_response_plot, 
    create_waterfall_plot)

from .SpatialModulationIndex import (
    double_gaussian, 
    fit_response_profile, 
    calculate_SMI, 
    calculate_SMI_BBBB, 
    plot_SMI_results, 
    plot_SMI_results_BBBB, 
    analyze_spatial_modulation, 
    analyze_spatial_modulation_BBBB)

from .loadData import (
    dataLoader)

from .Preprocess import (
    preprocess_2pVR)

# Specify what is available when you import the package
__all__ = ["read_xml", "write_h5", "recursively_save_dict_contents_to_group", "recursively_load_dict_contents_from_group",
           "time2float", "time2str",
           "TwoP",
           "find_temporal_offset", "apply_temporal_offset", "smooth_spikes", "plot_comparison", "plot_sample_cells", "spatial_smooth",
           "reshape_into_laps", "process_data_with_trial_filtering",
           "spatial_assignment",
           "test_cell_reliability", "test_cell_reliability_with_edge_visualization", "plot_edge_activity_distributions", "visualize_cell_edge_profiles", "normalize_spatial_activity", "plot_reliable_cells_side_by_side", "plot_reliable_cells_grid","plot_reliable_cells_waterfall", "evaluate_pattern_similarity", "combined_reliability_test", 
           "create_response_plot", "create_waterfall_plot",
           "double_gaussian", "fit_response_profile", "calculate_SMI", "calculate_SMI_BBBB", "plot_SMI_results", "plot_SMI_results_BBBB", "analyze_spatial_modulation", "analyze_spatial_modulation_BBBB",
           "dataLoader",
           "preprocess_2pVR"]