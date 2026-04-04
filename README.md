# An Agent-Based and Equation-Based Meta-Model of Behavioral Sink: The Case of Universe 25

Implementation of a dual formalization of population dynamics driven by density-dependent stress. The model is developed in two forms: a deterministic equation-based model (EBM) and a stochastic agent-based model (ABM).

## Files

* `Models.py`
  Core implementation of both models:

  * EBM (aggregate, deterministic)
  * ABM (individual-based, stochastic)

* `Fitting.py`
  Calibration of EBM parameters via genetic algorithm, minimizing NRMSE between simulated and empirical data.

* `ABM_Variation`
  Generates ABM simulations under parameter perturbations.
  For each parameter, 11 variations are explored from −10% to +10% (step 2%) to assess sensitivity of the resulting dynamics.

* `Results.ipynb`
  Loads simulation outputs from `ABM_Variation`, performs clustering on trajectories, and identifies dominant dynamic regimes.
  Visualization and post-processing of simulation outputs and clustering results.

* `Cluster_Analisis.ipynb`
  Exploratory analysis and validation of clustering structure.

* `best_fit/`
  Contains the optimal parameter set obtained from EBM calibration (GA), based on NRMSE minimization, along with associated outputs.

* `LICENSE`
  License of the repository.

## Requirements

Python 3.x
numpy, pandas, matplotlib, scikit-learn

## Usage

Run the pipeline in the following order:

python ABM_Variation

Then execute:

Cluster_Analisis.ipynb
Results.ipynb

## Notes

Calibration is performed on the EBM only, due to its deterministic structure.
The same parameter set is then used in the ABM.
ABM results are analyzed via clustering to identify dominant regimes.
Ensemble averages are not used when multiple regimes coexist.

## Reproducibility

All results can be reproduced by executing the scripts in sequence using the provided data and parameter configurations.
