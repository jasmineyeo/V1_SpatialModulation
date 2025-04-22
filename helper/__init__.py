# V1_SpatialModulation/
# ├── helper/
#     ├── __init__.py
#     ├── read_xml.py
#     ├── time2float.py    
#     ├── twop.py
#     ├── SpikeSmoothing.py
#     ├── spatial_discretization.py
#     ├── ReliabilityTesting.py
#     ├── ResponseVisualization.py
#     ├── BehavioralDataFiltering.py

# Import functions
from .read_xml import read_xml
from .time2float import time2float
from .twop import TwoP
from .SpikeSmoothing import apply_temporal_offset, smooth_spikes, plot_comparison, plot_sample_cells, spatial_smooth
from .BehavioralDataFiltering import reshape_into_laps, process_data_with_trial_filtering
from .spatial_discretization import spatial_assignment
from .ReliabilityTesting import test_cell_reliability, test_cell_reliability_with_edge_visualization, plot_edge_activity_distributions, visualize_cell_edge_profiles, normalize_spatial_activity, plot_reliable_cells_side_by_side, plot_reliable_cells_grid, plot_reliable_cells_waterfall 
from .ResponseVisualization import create_response_plot, create_waterfall_plot
from .SpatialModulationIndex import double_gaussian, fit_response_profile, calculate_SMI, plot_SMI_results, analyze_spatial_modulation

# Specify what is available when you import the package
__all__ = ["read_xml",
           "time2float",
           "TwoP",
           "apply_temporal_offset", "smooth_spikes", "plot_comparison", "plot_sample_cells", "spatial_smooth",
           "reshape_into_laps", "process_data_with_trial_filtering",
           "spatial_assignment",
           "test_cell_reliability", "test_cell_reliability_with_edge_visualization", "plot_edge_activity_distributions", "visualize_cell_edge_profiles", "normalize_spatial_activity", "plot_reliable_cells_side_by_side", "plot_reliable_cells_grid","plot_reliable_cells_waterfall",
           "create_response_plot", "create_waterfall_plot",
           "double_gaussian", "fit_response_profile", "calculate_SMI", "plot_SMI_results", "analyze_spatial_modulation"]