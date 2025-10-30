"""
Functions helping preprocessing and analysis for laminar spatial modulation
JSY, 2025
"""

# Import base modules
from .BehavioralDataFiltering import (
    calculate_speed_per_lap,
    reshape_into_laps_forward_only,
    process_data_with_speed_filtering,
    plot_speed_distribution
)
from .detrendAdaptation import (
    detrendAdaptation)

from .files import (
    read_xml,
    write_h5,
    read_h5,
    recursively_save_dict_contents_to_group,
    recursively_load_dict_contents_from_group,
)

from .loadData import (
    dataLoader)

from .Preprocess import (
    preprocess_2pVR)

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
    combined_reliability_test,
    combined_reliability_test_improved,
    find_robust_peak,
    evaluate_pattern_similarity_improved,
    test_cell_reliability_improved,
    improved_activity_threshold_check,
    plot_individual_reliable_cells,
    create_summary_figure,
    save_all_reliable_cell_plots)

from .ResponseVisualization import (
    create_response_plot, 
    create_waterfall_plot)

from .SpatialDiscretization import (
    spatial_assignment,
    spatial_assignment_with_physical_units)

from .SpatialModulationIndex import (
    double_gaussian, 
    fit_response_profile, 
    calculate_SMI_improved,
    plot_SMI_results_improved,
    analyze_spatial_modulation_improved,
    calculate_SMI_improved_debug)

from .SpeedTuningAnalysis import(
    SpeedTuningAnalysis
)

from .SpikeSmoothing import (
    calculate_sparsity_index,
    calculate_spatial_information,
    calculate_peak_to_baseline_ratio,
    apply_quality_filters,
    calculate_sharpness_metrics_for_offset,
    find_optimal_temporal_offset,
    create_offset_comparison_plot,
    run_offset_optimization,
    create_simple_before_after_comparison,
    find_best_example_cells,
    create_multiple_examples_split,
    create_five_detailed_examples,
    demonstrate_simple_offset_effect,
    apply_temporal_offset, 
    smooth_spikes, 
    plot_comparison, 
    plot_sample_cells, 
    spatial_smooth)

from .time import (
    time2float,
    time2str)

from .twop import (
    TwoP)


# Specify what is available when you import the package
__all__ = ["calculate_speed_per_lap", "reshape_into_laps_forward_only", "process_data_with_speed_filtering", "plot_speed_distribution",
           "detrendAdaptation",
            "read_xml", "write_h5", "read_h5", "recursively_save_dict_contents_to_group", "recursively_load_dict_contents_from_group",
            "dataLoader",
            "preprocess_2pVR",
            "test_cell_reliability", "test_cell_reliability_with_edge_visualization", "plot_edge_activity_distributions", "visualize_cell_edge_profiles", "normalize_spatial_activity", "plot_reliable_cells_side_by_side", "plot_reliable_cells_grid","plot_reliable_cells_waterfall", "evaluate_pattern_similarity", "combined_reliability_test", "combined_reliability_test_improved", "find_robust_peak", "evaluate_pattern_similarity_improved", "test_cell_reliability_improved", "improved_activity_threshold_check", "plot_individual_reliable_cells", "create_summary_figure", "save_all_reliable_cell_plots",
            "create_response_plot", "create_waterfall_plot",
            "spatial_assignment", "spatial_assignment_with_physical_units",
            "double_gaussian", "fit_response_profile", "calculate_SMI_improved", "plot_SMI_results_improved", "analyze_spatial_modulation_improved", "calculate_SMI_improved_debug",
            "SpeedTuningAnalysis",
            "calculate_sparsity_index", "calculate_spatial_information", "calculate_peak_to_baseline_ratio", "apply_quality_filters", "calculate_sharpness_metrics_for_offset", "find_optimal_temporal_offset", "create_offset_comparison_plot", "run_offset_optimization", "create_simple_before_after_comparison", "find_best_example_cells", "create_multiple_examples_split", "create_five_detailed_examples", "demonstrate_simple_offset_effect", "apply_temporal_offset", "smooth_spikes", "plot_comparison", "plot_sample_cells", "spatial_smooth",
            "time2float", "time2str",
            "TwoP"]