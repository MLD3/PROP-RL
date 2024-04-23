# OfflineRL_Pipeline
This repository contains the source code to replicate the pipeline described in "Optimizing Loop Diuretic Treatment in Hospitalized Patients: A Case Study in Practical Application of Offline Reinforcement Learning to Healthcare".

## Usage
- Refer to `requirements.txt` for the necessary conda packages.
- **data**: This directory contains sample input data.
- **pipeline**: This directory contains the offline RL pipeline for tabular data.
  - `1_run_kmeans.py`: Takes the embedded EHR features (`data/embedded.p`) and uses ensemble k-means clustering to discretize the data.
  - `2_run_environment.py`: Creates the transition and reward matrices from the training set.
  - `3_run_train_policy.py`: Trains all policies.
  - `4_run_evaluation.py`: Evaluates the learned policies on the validation and test sets.
  - `pipeline_example.ipynb`: Pseudocode for full pipeline. 

## Sample Input
An example usage of the pipeline is provided in `pipeline/pipeline_example.ipynb` with sample input data. The input are formatted as follows:
- `data/embedded.p`: Embedding of the raw feature values of the trajectories.
  - hosp_id: Unique ID of the trajectory/hospitalization
  - window_id: ID of each window
- `data/ens.csv`: Trajectory data with state, action, reward, and next state.
  - hosp_id, window_id: Same as above
  - death: Encoding for final outcome
  - For the last window in each trajectory, we assume that all actions (in our example, binary actions) will lead to a terminal absorbing state. The terminal state is defined by the final outcome (death). Thus the last window is recorded twice in the trajectory data. 
