# An Agent-Based and Equation-Based Meta-Model of Behavioral Sink: The Case of Universe 25

Code and data for a dual formalization of population dynamics driven by density-dependent stress, implemented as a deterministic equation-based model (EBM) and a stochastic agent-based model (ABM) that share the same parameters.

## Parameter naming

In the code, the density-to-stress amplification parameter is called `c`. In the paper it is called `p`. All other parameters (`m`, `n`, `d`, `k`, `R`) have the same name in code and paper.

## Files

| File | Content |
|---|---|
| `Models.py` | EBM (`aggregate_model`) and ABM (`IndividualModel`) |
| `Fitting.py` | Calibration of the EBM with a genetic algorithm (NRMSE minimization). Writes `best_fit_ebm.csv` |
| `best_fit_ebm.csv` | Calibrated parameters (initial population P_0 = 8) |
| `Equilibrium_table.py` | Fixed points of the EBM, their stability, and the equilibrium as a function of `d` (Table 4). Writes `equilibrium_table.csv` |
| `ABM_Variation` | ABM simulations under ±10% parameter perturbations (step 2%), 1000 runs each. Writes `Universe_25_<param>_<m|p><pct>.csv` |
| `Cluster_Analisis.ipynb` | Trajectory features, HDBSCAN clustering, modal cluster and NRMSE |
| `Figures_PLOS.py` | Figures 2–4 of the paper (TIFF, PLOS format) |
| `Results.ipynb`, `Equilibria.ipynb` | Exploratory analyses and plots |

### Data provided

| File | Produced by | Used for |
|---|---|---|
| `clusters.csv` | `Cluster_Analisis.ipynb` | Fig 3, Fig 4 |
| `cluster_peak_mapping.csv` | `Cluster_Analisis.ipynb` | Fig 3, Fig 4 |
| `modal_cluster_mean_timeseries.csv` | `Cluster_Analisis.ipynb` | NRMSE of the modal ABM trajectory |
| `NRMSE_param_variation.csv` | `Cluster_Analisis.ipynb` | Fig 2, Table 3 |

The raw ABM outputs (`Universe_25_*.csv`, `all_data.csv`, about 9 GB) are not included because of their size. They can be regenerated with `ABM_Variation`. Since the ABM is stochastic, regenerated outputs are statistically equivalent but not identical to the provided ones.

The empirical Universe 25 time series, digitized from Calhoun's published figures, is defined directly in the scripts (`t_ED`, `P_ED`).

## Usage

```
pip install -r requirements.txt

python Fitting.py               # EBM calibration -> best_fit_ebm.csv
python Equilibrium_table.py     # fixed-point analysis -> Table 4
python ABM_Variation            # ABM simulations (several hours)
# run Cluster_Analisis.ipynb    # clustering and NRMSE -> csv files above
python Figures_PLOS.py          # Figures 2-4
```

## License

MIT License, see `LICENSE`.
