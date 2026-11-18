import numpy as np
import pandas as pd
from Models import aggregate_model

df = pd.read_csv("best_fit_ebm.csv")
p, m, n, d, k, R = df.loc[0, ["c", "m", "n", "d", "k", "R"]]

P_0, S_0 = 8, 0
T_fit, T_long = 2000, 20000
t_ED = np.array([0, 80, 315, 560, 736, 800, 900, 1000, 1280, 1350, 1480])
P_ED = np.array([8, 20, 620, 2200, 2056, 1800, 1500, 1250, 680, 320, 95], dtype=float)

S_star = (n - m) / (k * m)


def P_star(d_):
    return R * S_star * d_ / ((1 - d_) * p * (1 + S_star))


def d_threshold():
    a = p * (1 + S_star) / (R * S_star)
    return a / (1 + a)


def eigenvalues(d_):
    P, S = P_star(d_), S_star
    S1 = (S + p * (P / R) * (1 + S)) * (1 - d_)
    dS1_dP = p / R * (1 + S) * (1 - d_)
    dS1_dS = (1 + p * P / R) * (1 - d_)
    B = n / (1 + k * S1)
    dB_dS1 = -n * k / (1 + k * S1) ** 2
    J = np.array([
        [1 + B - m + P * dB_dS1 * dS1_dP, P * dB_dS1 * dS1_dS],
        [dS1_dP, dS1_dS],
    ])
    return np.linalg.eigvals(J)


def nrmse(P_ts):
    P_model = np.asarray(P_ts)[t_ED]
    return np.sqrt(np.mean((P_model - P_ED) ** 2)) / (P_ED.max() - P_ED.min())


def day_below_one(P_ts):
    below = np.where(np.asarray(P_ts) < 1)[0]
    return int(below[0]) if len(below) else None


print(f"Extinction (0, 0): eigenvalues {1 + n - m:.6f}, {1 - d:.6f}")
print(f"Interior: S* = {S_star:.4f}, P* = {P_star(d):.2f}, |lambda| = {np.abs(eigenvalues(d)).max():.8f}")
print(f"d threshold for P* = 1: {d_threshold():.3e}")

rows = []
for d_ in [1e-6, d_threshold(), 1e-5, d, 1e-4, 3e-4, 1e-3]:
    _, P_ts = aggregate_model(T_long, p, m, n, d_, k, R, P_0, S_0)
    rows.append({
        "d": d_,
        "P_star": P_star(d_),
        "NRMSE": nrmse(P_ts[:T_fit + 1]),
        "day_P_below_1": day_below_one(P_ts),
    })

table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv("equilibrium_table.csv", index=False)
