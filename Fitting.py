from Models import aggregate_model
import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt

T = 2000
P_0 = 10
s_0 = 0

t_ED = np.array([0, 80, 315, 560, 736, 800, 900, 1000, 1280, 1350, 1480], dtype=int)
p_ED = np.array([8, 20, 620, 2200, 2056, 1800, 1500, 1250, 680, 320, 95], dtype=float)

def nrmse_on_points(M_ts, t, y):

    M_ED = np.asarray(M_ts, float)[np.asarray(t, int)]
    y = np.asarray(y, float)

    rmse = float(np.sqrt(np.mean((M_ED - y) ** 2)))
    return rmse / (y.max() - y.min())

def objective_function(params):
    c_, m_, n_, d_, k_, R_ = params
    _, P_ts_ = aggregate_model(T, c_, m_, n_, d_, k_, R_, P_0, s_0)
    return nrmse_on_points(P_ts_, t_ED, p_ED)

varbound = np.array([
    [5.10e-05, 5.11e-04], #c
    [5.20e-03, 5.30e-03], #m
    [1.92e-02, 1.93e-02], #n
    [2.83e-08, 2.84e-08], #d
    [3.52e+01, 3.53e+01], #k
    [3.80e+03, 3.80e+03], #R
], dtype=float)

algorithm_param = {
    'max_num_iteration': 1200,
    'population_size': 500,
    'mutation_probability': 0.2,
    'elit_ratio': 0.05,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': 250
}

model = ga(
    function=objective_function,
    dimension=6,
    variable_type='real',
    variable_boundaries=varbound,
    algorithm_parameters=algorithm_param
)

model.run()

best_params = model.output_dict['variable']
best_objective = model.output_dict['function']

_, P_ts_best = aggregate_model(T, *best_params, P_0, s_0)
rmse = nrmse_on_points(P_ts_best, t_ED, p_ED)

print(f'Best Parameters: {best_params}')
print(f'Best Objective: {best_objective}')
print(f'RMSE: {rmse}')

results_df = pd.DataFrame([{
    "c": best_params[0],
    "m": best_params[1],
    "n": best_params[2],
    "d": best_params[3],
    "k": best_params[4],
    "R": best_params[5],
    "RMSE_EBM": rmse
}])

results_df.to_csv("best_fit_ebm.csv", index=False)