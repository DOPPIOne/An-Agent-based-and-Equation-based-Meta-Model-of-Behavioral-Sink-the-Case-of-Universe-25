import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline
from PIL import Image

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

LABEL = {"c": "p"}
DPI = 300

T = 2000
P_0 = 8
s_0 = 0
t_ED = np.array([0, 80, 315, 560, 736, 800, 900, 1000, 1280, 1350, 1480])
P_ED = np.array([8, 20, 620, 2200, 2056, 1800, 1500, 1250, 680, 320, 95], dtype=float)


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    Image.open(OUT / name).convert("RGB").save(OUT / name, compression="tiff_lzw", dpi=(DPI, DPI))


def aggregate_model(T, p, m, n, d, k, R, P_0, S_0):
    P, S = P_0, S_0
    P_ts = [P_0]
    for _ in range(T):
        S = (S + p * (P / R) * (1 + S)) * (1 - d)
        P = max(P * (1 + n / (1 + k * S) - m), 0)
        P_ts.append(P)
    return np.asarray(P_ts, float)


def remap_clusters(clusters, mapping):
    df = clusters.merge(mapping, how="left",
                        left_on=["param", "variation", "cluster"],
                        right_on=["param", "variation", "old_cluster"])
    df["cluster_mapped"] = np.where(df["cluster"] == -1, -1, df["new_cluster"])
    return df


# =========================================================
# Fig 2 - robustness profiles
# =========================================================
def fig_robustness(nrmse):
    params = ["R", "c", "d", "k", "m", "n"]
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.2), sharex=True)

    for ax, param in zip(axes.flat, params):
        sub = nrmse[nrmse["parameter"] == param].sort_values("variation")
        x = sub["variation"].to_numpy(float)
        y = sub["NRMSE"].to_numpy(float)
        y = y - y[x == 0][0]

        xs = np.linspace(x.min(), x.max(), 300)
        ax.plot(xs, make_interp_spline(x, y, k=3)(xs), color="navy", lw=2, alpha=0.75)
        ax.scatter(x[x != 0], y[x != 0], s=18, color="navy", alpha=0.9)
        ax.scatter(x[x == 0], y[x == 0], s=18, color="salmon", zorder=5)
        ax.axvline(0, ls="--", lw=0.8, color="gray")
        ax.axhline(0, ls=":", lw=0.8, color="gray")
        ax.set_title(LABEL.get(param, param))
        ax.set_xticks(x[::2])
        ax.set_xticklabels([f"{int(v)}%" for v in x[::2]])

    fig.supxlabel("Variation (%)", fontsize=9)
    fig.supylabel("ΔNRMSE", fontsize=9)
    fig.tight_layout()
    save(fig, "Fig2.tif")


# =========================================================
# Fig 3 - cluster composition
# =========================================================
def fig_clusters(clusters):
    df = clusters[clusters["cluster_mapped"] != -1]
    counts = df.groupby(["param", "variation", "cluster_mapped"]).size().reset_index(name="count")
    counts["perc"] = 100 * counts["count"] / counts.groupby(["param", "variation"])["count"].transform("sum")
    summary = counts.groupby(["param", "cluster_mapped"])["perc"].agg(["mean", "std"]).reset_index()

    param_order = sorted(summary["param"].unique())
    cluster_order = sorted(summary["cluster_mapped"].unique())
    colors = plt.cm.cividis(np.linspace(0.2, 0.9, len(cluster_order)))

    x = np.arange(len(param_order))
    width = 0.8 / len(cluster_order)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))

    for i, cl in enumerate(cluster_order):
        sub = summary[summary["cluster_mapped"] == cl].set_index("param").reindex(param_order)
        ax.bar(x - 0.4 + width / 2 + i * width, sub["mean"], width=width, color=colors[i],
               yerr=sub["std"].fillna(0),
               error_kw={"ecolor": "salmon", "elinewidth": 1.2, "capsize": 3, "capthick": 1.2})

    y_max = 5 * math.ceil((summary["mean"] + summary["std"].fillna(0)).max() / 5 + 1)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, 10))
    ax.set_yticklabels([f"{int(v)}%" for v in np.arange(0, y_max + 1, 10)])
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(p, p) for p in param_order])
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Percentage")
    ax.grid(axis="y", ls="--", alpha=0.35)

    handles = [Line2D([0], [0], color=colors[i], lw=6, label=f"Cluster {int(cl)}")
               for i, cl in enumerate(cluster_order)]
    handles.append(Line2D([0], [0], color="salmon", lw=2, label="Standard deviation"))
    ax.legend(handles=handles, frameon=False, loc="upper left", ncol=4)

    fig.tight_layout()
    save(fig, "Fig3.tif")


# =========================================================
# Fig 4 - ABM vs EBM
# =========================================================
def fig_abm_vs_ebm(nrmse, clusters, fit):
    best = nrmse[nrmse["variation"] == 0].sort_values("NRMSE").iloc[0]
    param = best["parameter"]

    runs = pd.read_csv(DATA / f"Universe_25_{param}_p00.csv", usecols=["id", "t", "population"])
    pivot = runs.pivot(index="t", columns="id", values="population").sort_index()
    t = pivot.index.to_numpy(float)
    ids = pivot.columns.to_numpy()
    Y = pivot.to_numpy(float)

    labs = clusters[(clusters["param"] == param) & (clusters["variation"] == 0)].set_index("id")
    cluster_old = labs["cluster"].reindex(ids).fillna(-1).to_numpy()
    cluster_new = labs["cluster_mapped"].reindex(ids).fillna(-1).to_numpy()

    valid = cluster_old[cluster_old != -1]
    modal = pd.Series(valid).value_counts().idxmax()
    mean = Y[:, cluster_old == modal].mean(axis=1)

    clust_ids = sorted(set(cluster_new) - {-1})
    colors = {c: plt.cm.cividis(v) for c, v in zip(clust_ids, np.linspace(0.15, 0.82, len(clust_ids)))}

    P_ebm = aggregate_model(T, *fit, P_0, s_0)
    y_lim = 1.05 * max(np.nanmax(Y), P_ebm.max(), P_ED.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True)

    for j in range(Y.shape[1]):
        cl = cluster_new[j]
        if cl == -1:
            ax1.plot(t, Y[:, j], color="lightgrey", alpha=0.01, lw=0.5)
        else:
            ax1.plot(t, Y[:, j], color=colors[cl], alpha=0.1, lw=0.5)

    for cl in clust_ids:
        ax1.plot([], [], color=colors[cl], lw=1.5, label=f"Cluster {int(cl)}")
    ax1.plot(t, mean, color="navy", lw=2.5, label="Modal mean")
    ax1.scatter(t_ED, P_ED, color="salmon", s=20, zorder=5, label="Empirical data")
    ax1.set_title("ABM")
    ax1.set_xlabel("t (days)")
    ax1.set_ylabel("Population")
    ax1.set_ylim(0, y_lim)
    ax1.legend(frameon=False)

    ax2.plot(t, mean, color="skyblue", ls="--", lw=2, label="ABM modal mean")
    ax2.plot(np.arange(len(P_ebm)), P_ebm, color="navy", lw=2, label="EBM")
    ax2.scatter(t_ED, P_ED, color="salmon", s=20, zorder=5, label="Empirical data")
    ax2.set_title("ABM vs EBM")
    ax2.set_xlabel("t (days)")
    ax2.legend(frameon=False)

    fig.tight_layout()
    save(fig, "Fig4.tif")


nrmse = pd.read_csv(DATA / "NRMSE_param_variation.csv")
clusters = remap_clusters(pd.read_csv(DATA / "clusters.csv"),
                          pd.read_csv(DATA / "cluster_peak_mapping.csv"))
fit = pd.read_csv(DATA / "best_fit_ebm.csv").loc[0, ["c", "m", "n", "d", "k", "R"]].to_numpy(float)

fig_robustness(nrmse)
fig_clusters(clusters)
fig_abm_vs_ebm(nrmse, clusters, fit)
