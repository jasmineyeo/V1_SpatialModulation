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
├── Preprocess_MultiRecordings.py  # Preprocessing script for sessions with multiple recordings
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

There are two parallel sub-pipelines: 
**Standard SMI** (full-session SMI per day) and **Within-Session SMI** (how SMI develops over the course of a single session).

---

#### Pipeline A: Full-Session SMI (one SMI value per session, tracks learning across days)

```
Raw 2p data  →  Preprocess.py / Preprocess_MultipleRecordings.py
                        │
                        │  *preproc*.h5
                        ▼
        ┌───────────────────────────────────────────┐
        │  helper/SMICalculation_LayerSpecific_      │
        │           SingleRecording.py               │
        │  (Run_SMI_Layer_Analysis /                 │
        │   Run_SMI_AxonalImaging_window_Analysis)   │
        └─────────────────┬─────────────────────────┘
                          │  called by ↓
                          ▼
        SMI_FullSession_Batch.py
                          │  *_smi_results.h5  (one per session)
                          ▼
        SMI_FullSession_WithinAnimal.py
                          │  one animal, across days
                          ▼
        SMI_FullSession_AcrossAnimals.py          ← final population analysis
```

##### `helper/SMICalculation_LayerSpecific_SingleRecording.py`
The core per-session analysis module. Not run directly — called by the batch script.
- **Input:** Session folder containing `*preproc*.h5`
- **Key functions:**
  - `Run_SMI_Layer_Analysis()` — for prism (layered) recordings: applies onset/reward zone filtering, computes SMI per layer (L2/3, L4, L5, L6), plots sorted response maps and layer distributions
  - `Run_SMI_AxonalImaging_window_Analysis()` — for window/axonal imaging sessions: same pipeline adapted for non-layered data
- **Output:** `{date}_{animal}_smi_results.h5` saved in the session folder

##### `SMI_FullSession_Batch.py`
Batch runner — loops over a user-defined list of session folders and calls the single-recording analysis on each.
- **Input:** List of TSeries session directories (edit `session_dirs` at top of script). Each must contain a `*preproc*.h5` file.
- **Output:** One `*_smi_results.h5` per session, saved in each session folder
- **Note:** Update the glob pattern `"*preproc*.h5"` (with wildcard before `.h5`) so it finds both `*_preproc.h5` and `*_preproc_multi.h5` outputs

##### `SMI_FullSession_WithinAnimal.py`
Loads all `*_smi_results.h5` files for one animal and compares SMI across recording days.
- **Input:** Animal's root directory (searched recursively for `*_smi_results.h5`)
- **Analyses:**
  - Temporal progression of median SMI per layer (Days 1–N)
  - Early vs. late comparison (Days 1–2 vs. Days 6–7)
  - Proportion of spatially modulated cells (SMI > 0.1) per layer per day
  - Gap closure: do L2/3 cells catch up to deeper layers over time?
- **Output:** Stability plots, layer progression figures, significance markers

##### `SMI_FullSession_AcrossAnimals.py`
Loads `*_smi_results.h5` from all animals under a shared parent directory and runs pooled population-level analysis.
- **Input:** Parent directory (e.g. `D:\V1_SpatialModulation\2p\V1_prism\`), searched recursively
- **Analyses:**
  1. Day 1 layer differences — do deeper layers start with higher SMI?
  2. Temporal progression per layer (pooled) — does SMI increase over days?
  3. Layer development rate comparisons — do layers develop at different rates?
  4. Early (Days 1–2) vs. Late (Days 6–7) comparison (pooled)
  5. Gap closure — does the Deep − Superficial SMI gap narrow over time?
  6. Individual animal trajectory tracking
  7. 3×3 publication-ready summary figure + summary tables
- **Statistics:** Cliff's Delta, Kruskal-Wallis + Mann-Whitney + FDR correction, permutation tests (slope, slope comparison, group differences), bootstrap confidence intervals
- **Output:** Population statistics figures, `smi_summary_table.csv`, `smi_slopes_table.csv`

---

#### Pipeline B: Lap-Chunk SMI (SMI computed in lap blocks within a session, tracks intra-session convergence)

Asks: Does SMI increase over the course of a single recording? How many laps are needed to plateau? Do deeper layers stabilize faster?

```
*preproc*.h5  (single session)
      │
      ▼
SMI_LapChunk_SingleRecording.py
      │  *_within_session_smi.h5  (per session)
      ▼
SMI_LapChunk_WithinAnimal.py
      │  one animal, across days
      │  run_across_days_analysis_revised()
      ▼
SMI_LapChunk_AcrossAnimals.py
      │  run_across_animals_analysis_revised()
      ▼
      Population-level lap-chunk figures
```

##### `SMI_LapChunk_SingleRecording.py`
Runs the lap-chunk SMI analysis on a single recording session.
- **Input:** Session folder containing `*preproc*.h5`
- **Analysis approaches:**
  - *Fixed chunks:* non-overlapping blocks of 20 laps — tracks SMI stability within a chunk
  - *Cumulative:* progressively adds laps (laps 1–20, 1–40, 1–60, ...) — tracks SMI convergence
- **Output:** `*_within_session_smi.h5` saved in the session folder; figures showing SMI vs. lap chunk per layer

##### `SMI_LapChunk_WithinAnimal.py`
Compares lap-chunk SMI curves across days for one animal.
- **Input:** Animal directory (searched recursively for `*_within_session_smi.h5`)
- **Key function:** `run_across_days_analysis_revised()`
- **Output:** Per-animal across-day lap-chunk summaries

##### `SMI_LapChunk_AcrossAnimals.py`
Pools lap-chunk data across all animals for population-level figures.
- **Input:** List of `(animal_id, animal_dir)` pairs
- **Key function:** `run_across_animals_analysis_revised()`
- **Output:** Population-level lap-chunk figures

---

#### Deprecated Scripts

| Script | Status | Replaced by |
|--------|--------|-------------|
| `SMICalculation_LayerSpecific_CombinedTrialsWithinSession.py` | Superseded | `Preprocess_MultipleRecordings.py` + `SMI_FullSession_Batch.py` |

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
Raw Data
    |
Preprocess.py / Preprocess_MultipleRecordings.py  →  *_preproc*.h5
    |
SMI_FullSession_Batch.py                          →  *_smi_results.h5  (per session)
    |
SMI_FullSession_WithinAnimal.py                   →  stability plots (per animal)
    |
SMI_FullSession_AcrossAnimals.py                  →  population statistics
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

### Running Full-Session SMI Analysis
```bash
# 1. Preprocess data (one of:)
python Preprocess.py                      # single recording per session
python Preprocess_MultipleRecordings.py   # multiple recordings per session

# 2. Batch SMI analysis across all sessions
#    Edit session_dirs list in the script first
python 1.SpatialModulation_analysis/SMI_FullSession_Batch.py

# 3. Compare SMI across days within one animal
python 1.SpatialModulation_analysis/SMI_FullSession_WithinAnimal.py

# 4. Cross-animal population analysis
python 1.SpatialModulation_analysis/SMI_FullSession_AcrossAnimals.py
```

### Running Lap-Chunk SMI Analysis
```bash
# 1. Run per session (edit data_filepath in script)
python 1.SpatialModulation_analysis/SMI_LapChunk_SingleRecording.py

# 2. Compare across days for one animal (edit animal_dir in script)
python 1.SpatialModulation_analysis/SMI_LapChunk_WithinAnimal.py

# 3. Population-level analysis across animals (edit animals list in script)
python 1.SpatialModulation_analysis/SMI_LapChunk_AcrossAnimals.py
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
