# Spatial Modulation Analysis Pipeline

**Project:** V1_SpatialModulation
**Location:** `SpatialModulation_analysis/`
**Author:** JSY
**Last Updated:** December 2025

---

## Table of Contents
- [Overview](#overview)
- [Pipeline Workflows](#pipeline-workflows)
- [File Reference](#file-reference)
- [Data Flow Diagrams](#data-flow-diagrams)
- [Quick Start Guide](#quick-start-guide)

---

## Overview

This directory contains analysis scripts for investigating spatial modulation in mouse V1 during navigation tasks. Main analysis categories:
1. **SMI (Spatial Modulation Index)** - Quantify place-cell-like activity
2. **Landmark Preference** - Identify landmark-responsive cells
3. **Speed Modulation** - Analyze speed tuning properties
4. **PCA** - Dimensionality reduction of spatial responses

---

## Pipeline Workflows

### 1. SMI Analysis Pipeline

**Purpose:** Calculate and compare spatial modulation indices across layers, sessions, and animals

#### Single Recording Analysis
```
SpatialModulation_analysis/SMICalculation_LayerSpecific_SingleRecording.py
```
- **Input:** Single recording session data (mat/h5 files)
- **Output:** Layer-specific SMI values, spatial tuning curves
- **Use case:** First-pass analysis of one recording
- **Next step:** Batch processing

#### Batch Processing
```
SpatialModulation_analysis/SMICalculation_LayerSpecific_Batch.py
```
- **Input:** Multiple recording sessions (defined in script)
- **Output:** Compiled SMI metrics across recordings
- **Use case:** Process all recordings for one animal
- **Dependencies:** SingleRecording script must work first

#### Within-Session Analysis (Combined Trials)
```
SpatialModulation_analysis/SMICalculation_LayerSpecific_CombinedTrialsWithinSession.py
```
- **Input:** Multiple trials from same session
- **Output:** Trial-averaged SMI values
- **Use case:** Improve SNR by combining trials
- **Note:** Alternative approach to single-trial analysis

#### Within-Session Workflow (Full Pipeline)
```
1. SpatialModulation_analysis/SMICalculation_LayerSpecific_WithinSession_SingleRecording.py
   └─> Single recording, within-session SMI

2. SpatialModulation_analysis/SMICalculation_LayerSpecific_WithinSession_Batch.py
   └─> Batch process multiple recordings (within-session mode)

3. SpatialModulation_analysis/SMICalculation_LayerSpecific_WithinSession_AcrossRecordings.py
   └─> Compare across recordings for one animal

4. SpatialModulation_analysis/SMICalculation_LayerSpecific_WithinSession_AcrossAnimals.py
   └─> Cross-animal comparison (within-session SMI)

5. SpatialModulation_analysis/SMICalculation_LayerSpecific_WithinSession_TEST_visualization.py
   └─> Quality control plots and visualization tests
```

#### Cross-Session & Cross-Animal Comparisons
```
SpatialModulation_analysis/SMICalculation_CompareSessionsWithinAnimal.py
```
- **Input:** Multiple sessions from same animal (batch output)
- **Output:** Session stability plots, correlation matrices
- **Use case:** Track SMI changes across days

```
SpatialModulation_analysis/SMI_CompareAcrossAnimals_pt1.py
SpatialModulation_analysis/SMI_CompareAcrossAnimals_pt2.py
```
- **Input:** SMI data from multiple animals
- **Output:** Population statistics, cross-animal summary figures
- **Use case:** Publication-ready group statistics
- **Note:** Pt1 = data loading/processing, Pt2 = statistical tests/plotting

```
SpatialModulation_analysis/SMICalculation_AnalyzeAcrossAnimals.py
```
- **Input:** Aggregated SMI data across all animals
- **Output:** Comprehensive cross-animal analysis
- **Use case:** Final summary analysis for all animals combined

---

### 2. Landmark Preference Analysis

**Purpose:** Identify cells responsive to visual landmarks in the VR corridor

```
1. SpatialModulation_analysis/LandmarkPrefernce_SingleSessionAnalysis.py
   └─> Input: Single session data
   └─> Output: Landmark-responsive cell identification, tuning curves
   └─> Use case: Initial exploration of landmark selectivity

2. SpatialModulation_analysis/LandmarkPreference_Batch.py
   └─> Input: Multiple sessions (batch config)
   └─> Output: Compiled landmark preference across recordings
   └─> Use case: Process all sessions efficiently

3. SpatialModulation_analysis/LandmarkPrefernce_CompareSessionsWithinAnimal.py
   └─> Input: Batch output from one animal
   └─> Output: Landmark preference stability across sessions
   └─> Use case: Within-animal consistency analysis

4. SpatialModulation_analysis/LandmarkPreference_AnalyzeAcrossAnimals.py
   └─> Input: All animals' landmark data
   └─> Output: Population statistics, layer-specific landmark tuning
   └─> Use case: Final cross-animal summary
```

---

### 3. Speed Modulation Analysis

**Purpose:** Quantify how running speed affects neural activity

```
SpatialModulation_analysis/SpeedModulation_Run_Speed_Tuning_by_Layer.py
```
- **Input:** Recording data with speed/position information
- **Output:** Speed tuning curves per layer
- **Use case:** Identify speed-modulated cells

```
SpatialModulation_analysis/SpeedModulation_Run_Speed_Tuning_by_Layer_Bins.py
```
- **Input:** Same as above
- **Output:** Binned speed analysis (discrete speed categories)
- **Use case:** Categorical speed comparisons (slow/medium/fast)

```
SpatialModulation_analysis/SpeedModulation_analyze_across_sessions.py
```
- **Input:** Multiple sessions
- **Output:** Cross-session speed modulation statistics
- **Use case:** Speed tuning stability over time

```
SpatialModulation_analysis/SpeedModulation_visualize_speed_modulation_analyze_across_Sessions.py
```
- **Input:** Cross-session speed data
- **Output:** Comprehensive visualization of speed effects
- **Use case:** Generate publication figures

---

### 4. PCA Pipeline

**Purpose:** Dimensionality reduction to identify population-level spatial coding patterns

```
Step 1: SpatialModulation_analysis/PCA_DataAggregation.py
        └─> Input: Raw spatial response matrices from all sessions
        └─> Output: {animal}_pca_data.h5 (N_cells × N_spatial_bins)
        └─> Function: Aggregate and normalize data across sessions

Step 2: SpatialModulation_analysis/PCA_DataVerification.py
        └─> Input: {animal}_pca_data.h5
        └─> Output: QC plots, data integrity checks
        └─> Function: Verify data quality before PCA

Step 3: SpatialModulation_analysis/PCA_Analysis.py
        └─> Input: {animal}_pca_data.h5
        └─> Output: PC loadings, scree plot, variance explained
        └─> Function: Perform PCA decomposition

Step 4: SpatialModulation_analysis/PCA_Diagnostics.py
        └─> Input: PCA results from Step 3
        └─> Output: Diagnostic plots (reconstruction error, etc.)
        └─> Function: Validate PCA quality

Step 5: SpatialModulation_analysis/PCA_Clustering.py
        └─> Input: PC scores from Step 3
        └─> Output: Cell clusters, cluster assignments
        └─> Function: K-means/hierarchical clustering in PC space

Step 6: SpatialModulation_analysis/PCA_Interpretation.py
        └─> Input: PC loadings, cluster assignments
        └─> Output: Biological interpretation plots
        └─> Function: Relate PCs to spatial/behavioral features

Step 7: SpatialModulation_analysis/PCA_LayerStatistics.py
        └─> Input: PCA results + layer labels
        └─> Output: Layer-specific PC statistics
        └─> Function: Compare PC distributions across layers

Step 8: SpatialModulation_analysis/PCA_ComprehensiveAnalysis.py
        └─> Input: All PCA outputs
        └─> Output: Combined analysis summary
        └─> Function: Generate final PCA report
```

---

## File Reference

### Analysis Scripts (by category)

| File | Lines | Purpose | Key Outputs |
|------|-------|---------|-------------|
| **SMI Analysis** |
| `SMICalculation_LayerSpecific_SingleRecording.py` | ~23K | Single recording SMI | SMI values, tuning curves |
| `SMICalculation_LayerSpecific_Batch.py` | ~10K | Batch SMI processing | Compiled SMI across sessions |
| `SMICalculation_LayerSpecific_CombinedTrialsWithinSession.py` | ~19K | Trial-averaged SMI | Trial-combined metrics |
| `SMICalculation_LayerSpecific_WithinSession_SingleRecording.py` | ~39K | Within-session single rec | Within-session SMI |
| `SMICalculation_LayerSpecific_WithinSession_Batch.py` | ~25K | Batch within-session | Multiple recordings |
| `SMICalculation_LayerSpecific_WithinSession_AcrossRecordings.py` | ~17K | Compare recordings | Recording comparisons |
| `SMICalculation_LayerSpecific_WithinSession_AcrossAnimals.py` | ~26K | Cross-animal (within) | Animal comparisons |
| `SMICalculation_LayerSpecific_WithinSession_TEST_visualization.py` | ~11K | QC visualizations | Test plots |
| `SMICalculation_CompareSessionsWithinAnimal.py` | ~28K | Session stability | Session correlations |
| `SMI_CompareAcrossAnimals_pt1.py` | ~22K | Cross-animal pt1 | Data aggregation |
| `SMI_CompareAcrossAnimals_pt2.py` | ~14K | Cross-animal pt2 | Statistics, plots |
| `SMICalculation_AnalyzeAcrossAnimals.py` | ~49K | Final cross-animal | Summary figures |
| **Landmark Preference** |
| `LandmarkPrefernce_SingleSessionAnalysis.py` | ~65K | Single session landmarks | Landmark-tuned cells |
| `LandmarkPreference_Batch.py` | ~13K | Batch landmarks | Compiled landmark data |
| `LandmarkPrefernce_CompareSessionsWithinAnimal.py` | ~38K | Landmark stability | Session comparisons |
| `LandmarkPreference_AnalyzeAcrossAnimals.py` | ~47K | Cross-animal landmarks | Population stats |
| **Speed Modulation** |
| `SpeedModulation_Run_Speed_Tuning_by_Layer.py` | ~25K | Layer speed tuning | Speed curves by layer |
| `SpeedModulation_Run_Speed_Tuning_by_Layer_Bins.py` | ~22K | Binned speed analysis | Categorical speed |
| `SpeedModulation_analyze_across_sessions.py` | ~14K | Cross-session speed | Speed stability |
| `SpeedModulation_visualize_speed_modulation_analyze_across_Sessions.py` | ~43K | Speed visualization | Publication figures |
| **PCA Pipeline** |
| `PCA_DataAggregation.py` | ~21K | Aggregate spatial data | _pca_data.h5 |
| `PCA_DataVerification.py` | ~25K | Verify data quality | QC reports |
| `PCA_Analysis.py` | ~28K | Run PCA | PC loadings, scree |
| `PCA_Diagnostics.py` | ~30K | PCA diagnostics | Error analysis |
| `PCA_Clustering.py` | ~25K | Cluster in PC space | Cell clusters |
| `PCA_Interpretation.py` | ~24K | Interpret PCs | Biological meaning |
| `PCA_LayerStatistics.py` | ~21K | Layer PC stats | Layer comparisons |
| `PCA_ComprehensiveAnalysis.py` | ~72K | Full PCA summary | Combined report |

### Notebooks (exploratory/legacy)

| File | Purpose |
|------|---------|
| `CombiningMultipleTrialsWithinSession.ipynb` | Prototype for trial-averaging approach |
| `Layer-Specific_SpatialMod_Analysis_includingdataprocessing_BBBB_experience.ipynb` | Initial SMI exploration |
| `Layer-Specific_SpatialMod_Analysis_includingdataprocessing_BBBB_forcandidacy.ipynb` | Candidacy exam analysis |
| `SpeedAnalysis_DuringTraversal.ipynb` | Speed analysis prototyping |

---

## Data Flow Diagrams

### Typical SMI Workflow
```
Raw Data (mat/h5)
    ↓
SingleRecording.py → SMI values for one session
    ↓
Batch.py → SMI across multiple sessions
    ↓
CompareSessionsWithinAnimal.py → Stability analysis
    ↓
CompareAcrossAnimals.py → Population statistics
```

### PCA Workflow
```
Session 1, 2, ..., N (spatial responses)
    ↓
DataAggregation.py → pca_data.h5
    ↓
DataVerification.py → QC passed?
    ↓
Analysis.py → PC1, PC2, ..., PC10
    ↓
Clustering.py → Cell groups
    ↓
Interpretation.py + LayerStatistics.py → Biological insights
```

---

## Quick Start Guide

### Running SMI Analysis (First Time)
```bash
# 1. Start with single recording
python SMICalculation_LayerSpecific_SingleRecording.py

# 2. If successful, batch process
python SMICalculation_LayerSpecific_Batch.py

# 3. Compare across sessions
python SMICalculation_CompareSessionsWithinAnimal.py

# 4. Cross-animal analysis
python SMI_CompareAcrossAnimals_pt1.py
python SMI_CompareAcrossAnimals_pt2.py
```

### Running PCA Pipeline
```bash
# Must run in order:
python PCA_DataAggregation.py      # Step 1
python PCA_DataVerification.py     # Step 2 (QC)
python PCA_Analysis.py             # Step 3
python PCA_Clustering.py           # Step 4
python PCA_Interpretation.py       # Step 5
```

### Key Configuration Files
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
- Located in `../helper/` directory
- Automatically added to sys.path in each script

---

## Notes

- **File naming convention:** `Analysis_Scope_Details.py`
  - Analysis = SMI, LandmarkPreference, SpeedModulation, PCA
  - Scope = SingleRecording, Batch, WithinSession, AcrossAnimals

- **Data paths:** Most scripts point to `D:\V1_SpatialModulation\2p\V1_prism\`
  - Update paths in CONFIGURATION section for your system

- **Output locations:** Figures typically saved to `{data_dir}/figures/`

---

## Contact

**Author:** JSY
**Project Repository:** `c:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation\`
