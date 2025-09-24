# TDS_measure_predictions_and_mediations

Running CPM

input files:
- connectivity matrices stored in 3D (node x node x subject)
- data to model (sub x measure)
- confounds (sub x variable)

How to run CPM:
- Done in matlab
- Run runCPM_allvars.m as a wrapper script that calls the function runCPM
- This will save a predictions.mat file containing prediction strength from each permutation, a null distribution of predictions on shuffled data, and network masks for each permutation/cross-validation fold

How to run other analyses used in the study and how to generate figures:
- The jupyter notebook 'Figures_code.ipynb' will conduct all main analyses done in the study. Just run each cell sequentially.
- Make sure that the filepaths load in the correct data. Dependency files (e.g., surface files) are included in this repository. The output data from CPM (network masks and prediction strength values) are stored here: {insert link to data}. Be sure to update all relevant filepaths in the notebook accordingly!
