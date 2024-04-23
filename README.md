# OfflineRL_Pipeline
Optimizing Loop Diuretic Treatment in Hospitalized Patients: A Case Study in Practical Application of Offline Reinforcement Learning to Healthcare

## Usage
- Refer to `requirements.txt` for the necessary pip packages.
- **data**: This directory contains sample input data.
- **pipeline**: This directory contains the offline RL pipeline for tabular data.
  - `1_run_kmeans.py`: Takes the embedded EHR features (`data/embedded.p`) and uses ensemble k-means clustering to discretize the data.
  - `2_run_environment.py`: Creates the transition and reward matrices from the training set.
  - `3_run_train_policy.py`: Trains all policies.
  - `4_run_evaluation.py`: Evaluates the learned policies on the validation and test sets. 

## Input
An example usage of the pipeline is provided with dummy input data. Please refer to the sample input files and descriptions below for formatting requirements. 

### Tabular Data

