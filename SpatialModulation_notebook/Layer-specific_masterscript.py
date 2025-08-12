import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from helper import TwoP, read_xml, time2float
from helper import SpikeSmoothing, BehavioralDataFiltering as DF, spatial_discretization as SD, ReliabilityTesting as RT, ResponseVisualization as RV, SpatialModulationIndex as SMI
from helper.SpatialModulationIndexLayerSpecific import SpatialModulationIndexLayerSpecific as SMI_Layer
from helper.detrendAdaptation import detrendAdaptation as DA

from helper.preprocessData import load_data

from matplotlib import rcParams
rcParams['legend.fontsize'] = 14
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 20
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14

twop_filepath = r'F:\2P\spmod\250811_JSY_JSY044_SpatialModulation_Day1\TSeries-08112025-1505-001'
vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_08112025_04-04-19.txt"


load_data(twop_filepath, vr_filepath)