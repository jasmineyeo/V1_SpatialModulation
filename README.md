# V1_SpatialModulation

Scripts for analyzing spatial modulation responses in mouse V1 during virtual reality navigation tasks.

**Author:** JSY | **Last Updated:** February 2026

---

## Quick Setup

```bash
# Install conda environment
conda env create -f environment.yml
conda activate JSY_SpatialMod

# Install twopTools module (run from repository root)
pip install -e .
```

---

## Table of Contents
- [Project Structure](#project-structure)
- [Analysis Overview](#analysis-overview)
- [Pipeline Workflows](#pipeline-workflows)
- [Quick Start](#quick-start)
- [File Reference](#file-reference)

---

## Project Structure

```
V1_SpatialModulation/
├── Preprocess.py                  # Main preprocessing script
├── helper/                        # Shared helper modules
├── .Debugging/                    # Debugging & validation scripts (hidden)
├── .VisualResponseAnalysis/       # Visual response analysis notebooks (hidden)
├── 1.SpatialModulation_analysis/  # SMI calculation & analysis
├── 2.SpeedAnalysis/               # Speed modulation analysis
├── 3.LandmarkPreference/          # Landmark preference analysis
├── 4.PCA/                         # Principal Component Analysis
├── 5.AxonalSM_analysis/           # Axonal spatial modulation analysis
├── datalist.txt                   # Data list configuration
└── environment.yml                # Conda environment file
```

---

## Analysis Overview

Five main analysis pipelines, organized by numbered directories:

1. **SMI (Spatial Modulation Index)** (`1.SpatialModulation_analysis/`) - Meausre spatial modulation index 
2. **Speed Modulation** (`2.SpeedAnalysis/`) - Analyze speed tuning properties
3. **Landmark Preference** (`3.LandmarkPreference/`) - Identify landmark preference
4. **PCA** (`4.PCA/`) - Dimensionality reduction of spatial responses
5. **Axonal SM** (`5.AxonalSM_analysis/`) - Axon identification and merging for spatial modulation

---

## Pipeline Workflows

### 1. SMI Analysis Pipeline

**Purpose:** Calculate and compare spatial modulation indices across layers, sessions, and animals

**Location:** `1.SpatialModulation_analysis/`

#### Single Recording Analysis
```
1.SpatialModulation_analysis/SMICalculation_LayerSpecific_SingleRecording.py
```
- **Input:** Single recording session data (mat/h5 files)
- **Output:** Layer-specific SMI values, spatial tuning curves
- **Use case:** First-pass analysis of one recording
- **Next step:** Batch processing

#### Batch Processing
```
1.SpatialModulation_analysis/SMICalculation_LayerSpecific_Batch.py
```
- **Input:** Multiple recording sessions (defined in script)
- **Output:** Compiled SMI metrics across recordings
- **Use case:** Process all recordings for one animal
- **Dependencies:** SingleRecording script must work first

#### Within-Session Analysis (Combined Trials)
```
1.SpatialModulation_analysis/SMICalculation_LayerSpecific_CombinedTrialsWithinSession.py
```
- **Input:** Multiple trials from same session
- **Output:** Trial-averaged SMI values
- **Use case:** Improve SNR by combining trials
- **Note:** Alternative approach to single-trial analysis

#### Within-Session Workflow (Full Pipeline)
```
1. 1.SpatialModulation_analysis/SMICalculation_LayerSpecific_WithinSession_SingleRecording.py
   └─> Single recording, within-session SMI

2. 1.SpatialModulation_analysis/SMICalculation_LayerSpecific_WithinSession_Batch.py
   └─> Batch process multiple recordings (within-session mode)

3. 1.SpatialModulation_analysis/SMICalculation_LayerSpecific_WithinSession_AcrossRecordings.py
   └─> Compare across recordings for one animal

4. 1.SpatialModulation_analysis/SMICalculation_LayerSpecific_WithinSession_AcrossAnimals.py
   └─> Cross-animal comparison (within-session SMI)
```

#### Cross-Session & Cross-Animal Comparisons
```
1.SpatialModulation_analysis/SMICalculation_CompareSessionsWithinAnimal.py
```
- **Input:** Multiple sessions from same animal (batch output)
- **Output:** Session stability plots, correlation matrices
- **Use case:** Track SMI changes across days

```
1.SpatialModulation_analysis/SMI_CompareAcrossAnimals_pt1.py
1.SpatialModulation_analysis/SMI_CompareAcrossAnimals_pt2.py
```
- **Input:** SMI data from multiple animals
- **Output:** Population statistics, cross-animal summary figures
- **Use case:** Publication-ready group statistics
- **Note:** Pt1 = data loading/processing, Pt2 = statistical tests/plotting

```
1.SpatialModulation_analysis/SMICalculation_AnalyzeAcrossAnimals.py
```
- **Input:** Aggregated SMI data across all animals
- **Output:** Comprehensive cross-animal analysis
- **Use case:** Final summary analysis for all animals combined

---

### 2. Speed Modulation Analysis

**Purpose:** Quantify how running speed affects neural activity

**Location:** `2.SpeedAnalysis/`

```
2.SpeedAnalysis/SpeedModulation_Run_Speed_Tuning_by_Layer.py
```
- **Input:** Recording data with speed/position information
- **Output:** Speed tuning curves per layer
- **Use case:** Identify speed-modulated cells

```
2.SpeedAnalysis/SpeedModulation_Run_Speed_Tuning_by_Layer_Bins.py
```
- **Input:** Same as above
- **Output:** Binned speed analysis (discrete speed categories)
- **Use case:** Categorical speed comparisons (slow/medium/fast)

```
2.SpeedAnalysis/SpeedModulation_analyze_across_sessions.py
```
- **Input:** Multiple sessions
- **Output:** Cross-session speed modulation statistics
- **Use case:** Speed tuning stability over time

```
2.SpeedAnalysis/SpeedModulation_visualize_speed_modulation_analyze_across_Sessions.py
```
- **Input:** Cross-session speed data
- **Output:** Comprehensive visualization of speed effects
- **Use case:** Generate publication figures

---

### 3. Landmark Preference Analysis

**Purpose:** Identify cells responsive to visual landmarks in the VR corridor

**Location:** `3.LandmarkPreference/`

```
1. 3.LandmarkPreference/LandmarkPrefernce_SingleSessionAnalysis.py
   └─> Input: Single session data
   └─> Output: Landmark-responsive cell identification, tuning curves
   └─> Use case: Initial exploration of landmark selectivity

2. 3.LandmarkPreference/LandmarkPreference_Batch.py
   └─> Input: Multiple sessions (batch config)
   └─> Output: Compiled landmark preference across recordings
   └─> Use case: Process all sessions efficiently

3. 3.LandmarkPreference/LandmarkPrefernce_CompareSessionsWithinAnimal.py
   └─> Input: Batch output from one animal
   └─> Output: Landmark preference stability across sessions
   └─> Use case: Within-animal consistency analysis

4. 3.LandmarkPreference/LandmarkPreference_AnalyzeAcrossAnimals.py
   └─> Input: All animals' landmark data
   └─> Output: Population statistics, layer-specific landmark tuning
   └─> Use case: Final cross-animal summary
```

---

### 4. PCA Pipeline

**Purpose:** Dimensionality reduction to identify population-level spatial coding patterns

**Location:** `4.PCA/`

```
Step 1: 4.PCA/PCA_DataAggregation.py
        └─> Input: Raw spatial response matrices from all sessions
        └─> Output: {animal}_pca_data.h5 (N_cells x N_spatial_bins)
        └─> Function: Aggregate and normalize data across sessions

Step 2: 4.PCA/PCA_DataVerification.py
        └─> Input: {animal}_pca_data.h5
        └─> Output: QC plots, data integrity checks
        └─> Function: Verify data quality before PCA

Step 3: 4.PCA/PCA_Analysis.py
        └─> Input: {animal}_pca_data.h5
        └─> Output: PC loadings, scree plot, variance explained
        └─> Function: Perform PCA decomposition

Step 4: 4.PCA/PCA_Diagnostics.py
        └─> Input: PCA results from Step 3
        └─> Output: Diagnostic plots (reconstruction error, etc.)
        └─> Function: Validate PCA quality

Step 5: 4.PCA/PCA_Clustering.py
        └─> Input: PC scores from Step 3
        └─> Output: Cell clusters, cluster assignments
        └─> Function: K-means/hierarchical clustering in PC space

Step 6: 4.PCA/PCA_Interpretation.py
        └─> Input: PC loadings, cluster assignments
        └─> Output: Biological interpretation plots
        └─> Function: Relate PCs to spatial/behavioral features

Step 7: 4.PCA/PCA_LayerStatistics.py
        └─> Input: PCA results + layer labels
        └─> Output: Layer-specific PC statistics
        └─> Function: Compare PC distributions across layers

Step 8: 4.PCA/PCA_ComprehensiveAnalysis.py
        └─> Input: All PCA outputs
        └─> Output: Combined analysis summary
        └─> Function: Generate final PCA report
```

#### Additional PCA Scripts
```
4.PCA/PCA_AllCells_LayerAnalysis.py        - PCA across all cells with layer breakdown
4.PCA/PCA_SessionCorrection.py             - Correct for session-to-session variability
4.PCA/PCA_LandmarkAlignedAnalysis.py       - PCA aligned to landmark positions
4.PCA/PCA_LandmarkAdaptationConfound_Test.py - Test for adaptation confounds
4.PCA/PCA_WithinLayer_SMI_Analysis.py      - SMI analysis within PCA-defined groups
```

#### PCA Notebooks
```
4.PCA/PCA_LandmarkAlignedAnalysis_averagetraces.ipynb  - Average trace visualization
4.PCA/PCA_LandmarkAlignedAnalysis_clustering.ipynb     - Clustering of landmark-aligned responses
```

---

### 5. Axonal SM Analysis

**Purpose:** Axon identification and merging pipelines for spatial modulation analysis

**Location:** `5.AxonalSM_analysis/notebook/`

```
5.AxonalSM_analysis/notebook/Axon_Identification_Suite2p.ipynb   - Identify axons in Suite2p output
5.AxonalSM_analysis/notebook/AxonMerge_Suite2p.ipynb             - Merge axonal ROIs (Suite2p)
5.AxonalSM_analysis/notebook/AxonMerge_MGpipeline.ipynb          - Merge axonal ROIs (MG pipeline)
```

---

## Notebooks (Exploratory)

**Location:** `.VisualResponseAnalysis/`, `1.SpatialModulation_analysis/`, `2.SpeedAnalysis/`, and `4.PCA/`

| File | Location | Purpose |
|------|----------|---------|
| `normalGC6s_spatialmapping_analysis.ipynb` | `.VisualResponseAnalysis/` | Spatial mapping with normal GCaMP6s |
| `prism_analysis_ontreadmill.ipynb` | `.VisualResponseAnalysis/` | Prism-based analysis on treadmill |
| `spatialmapping_analysis.ipynb` | `.VisualResponseAnalysis/` | General spatial mapping analysis |
| `touchofevil_analysis.ipynb` | `.VisualResponseAnalysis/` | Touch-of-evil paradigm analysis |
| `CombiningMultipleTrialsWithinSession.ipynb` | `1.SpatialModulation_analysis/` | Prototype for trial-averaging approach |
| `Layer-Specific_SpatialMod_..._experience.ipynb` | `1.SpatialModulation_analysis/` | Initial SMI exploration |
| `Layer-Specific_SpatialMod_..._forcandidacy.ipynb` | `1.SpatialModulation_analysis/` | Candidacy exam analysis |
| `SpeedAnalysis_DuringTraversal.ipynb` | `2.SpeedAnalysis/` | Speed analysis prototyping |

---

## Debugging & Validation

**Location:** `.Debugging/`

| File | Purpose |
|------|---------|
| `Spatial_Smoothing_Window_Test.py` | Test spatial smoothing parameters |
| `Spike_Smoothing_Window_Test.py` | Test spike smoothing parameters |
| `speed_position_validation_afterfiltering.py` | Validate speed/position after filtering |

---

## Helper Modules

**Location:** `helper/`

| Module | Purpose |
|--------|---------|
| `__init__.py` | Package initialization |
| `axons.py` | Axon identification and processing utilities |
| `BehavioralDataFiltering.py` | Filter behavioral data (speed, position) |
| `correlation.py` | Correlation analysis utilities |
| `detrendAdaptation.py` | Remove adaptation trends from neural data |
| `files.py` | File I/O utilities |
| `loadData.py` | Load experimental data (mat/h5 formats) |
| `ReliabilityTesting.py` | Split-half and trial-to-trial reliability |
| `ResponseVisualization.py` | Plotting functions for neural responses |
| `SpatialDiscretization.py` | Bin continuous position into spatial bins |
| `SpatialModulationIndex.py` | Core SMI calculation functions |
| `SpatialModulationIndexLayerSpecific.py` | Layer-specific SMI calculations |
| `SpeedTuningAnalysis.py` | Speed tuning curve analysis |
| `SpikeSmoothing.py` | Temporal smoothing of spike/calcium data |
| `tif_stack_to_video.py` | Convert TIF stacks to video format |
| `time.py` | Time-related utilities |
| `trim_video_opencv.py` | Trim videos using OpenCV |
| `twop.py` | Two-photon imaging utilities |

---

## Data Flow Diagrams

### Typical SMI Workflow
```
Raw Data (mat/h5)
    |
SingleRecording.py -> SMI values for one session
    |
Batch.py -> SMI across multiple sessions
    |
CompareSessionsWithinAnimal.py -> Stability analysis
    |
CompareAcrossAnimals.py -> Population statistics
```

### PCA Workflow
```
Session 1, 2, ..., N (spatial responses)
    |
DataAggregation.py -> pca_data.h5
    |
DataVerification.py -> QC passed?
    |
Analysis.py -> PC1, PC2, ..., PC10
    |
Clustering.py -> Cell groups
    |
Interpretation.py + LayerStatistics.py -> Biological insights
```

---

## Quick Start Guide

### Running SMI Analysis (First Time)
```bash
# 1. Start with single recording
python 1.SpatialModulation_analysis/SMICalculation_LayerSpecific_SingleRecording.py

# 2. If successful, batch process
python 1.SpatialModulation_analysis/SMICalculation_LayerSpecific_Batch.py

# 3. Compare across sessions
python 1.SpatialModulation_analysis/SMICalculation_CompareSessionsWithinAnimal.py

# 4. Cross-animal analysis
python 1.SpatialModulation_analysis/SMI_CompareAcrossAnimals_pt1.py
python 1.SpatialModulation_analysis/SMI_CompareAcrossAnimals_pt2.py
```

### Running PCA Pipeline
```bash
# Must run in order:
python 4.PCA/PCA_DataAggregation.py      # Step 1
python 4.PCA/PCA_DataVerification.py     # Step 2 (QC)
python 4.PCA/PCA_Analysis.py             # Step 3
python 4.PCA/PCA_Clustering.py           # Step 4
python 4.PCA/PCA_Interpretation.py       # Step 5
```

### Key Configuration
- Each script has a `CONFIGURATION` section at the top
- Update paths to match your data directory structure
- Common parameters: `BASE_DIR`, `ANIMAL_ID`, `SESSION_DATE`

---

## Dependencies

**Python Version:** 3.8+

**Required Packages:**
- numpy
- scipy
- matplotlib
- h5py
- sklearn (for PCA)
- seaborn (for visualization)

**Custom Modules:**
- Located in `helper/` directory
- Automatically added to sys.path in each script

---

## Notes

- **Directory naming convention:**
  - Dot-prefixed directories (`helper/`, `.Debugging/`, `.VisualResponseAnalysis/`) are hidden utility/exploratory folders
  - Numbered prefixes (`1.` through `5.`) indicate the main analysis pipelines:
  - `1.SpatialModulation_analysis/` - Core SMI analysis
  - `2.SpeedAnalysis/` - Speed modulation
  - `3.LandmarkPreference/` - Landmark selectivity
  - `4.PCA/` - Dimensionality reduction
  - `5.AxonalSM_analysis/` - Axonal spatial modulation

- **File naming convention:** `Analysis_Scope_Details.py`
  - Analysis = SMI, LandmarkPreference, SpeedModulation, PCA
  - Scope = SingleRecording, Batch, WithinSession, AcrossAnimals

- **Data paths:** Most scripts point to `D:\V1_SpatialModulation\2p\V1_prism\`
  - Update paths in CONFIGURATION section for your system

- **Output locations:** Figures typically saved to `{data_dir}/figures/`

---

## Contact

**Author:** Jasmine Yeo (jasmineyeo@ucsb.edu)
**Project Repository:** `c:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation\`
