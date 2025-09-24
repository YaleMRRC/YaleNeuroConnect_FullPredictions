# TDS_measure_predictions_and_mediations

**Instructions for running CPM:**

Required input files:
- Connectivity matrices saved as a 3-dimensional variable (node x node x subject)
- Behavioral data to model (subject x measure)
- Confounds (subject x confound)
  *These are all stored in '/CPM_input'. The connectivity matrices ('average_mats.mat') may need to be downloaded separately from the rest of the repository. The version downloaded with the repository's .zip file may not work due to the large size of this file.

How to run CPM:
- CPM scripts are written in MATLAB
- Run runCPM_allvars.m as a wrapper script that calls the function runCPM
- Input data for CPM are stored in '/CPM_input'. Be sure to update relevant filepaths in runCPM_allvars.m accordingly!
- This will save a predictions.mat file containing prediction strength from each permutation, a null distribution of predictions on shuffled data, and network masks for each permutation/cross-validation fold.

**How to run other analyses used in the study and how to generate figures:**
- The jupyter notebook 'Figures_code.ipynb' will conduct all main analyses done in the study. Just run each cell sequentially.
- Make sure that the filepaths load in the correct data. Dependency files (e.g., surface files) are included in this repository. The output data from CPM (network masks and prediction strength values) are stored in '/CPM_output' as 'symptom_predictions.mat' and 'cognitive_predictions.mat'. Be sure to update all relevant filepaths in the notebook accordingly!
