import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler


INPUT_DIR = Path(".")
OUTPUT_DIR = Path("clusters")
OUTPUT_DIR.mkdir(exist_ok=True)

CASE_NAME = "Universe_25"
PARAM_NAMES = ["c", "d", "k", "n", "m", "R"]

MIN_CLUSTER_SIZE = 30
MIN_SAMPLES = 15
CLUSTER_SELECTION_METHOD = "eom"

MAX_SIM = 1000
N_COLS = 4

OUTPUT_CSV = OUTPUT_DIR / "universe25_modal_cluster_nrmse_feature_hdbscan.csv"

t_emp = np.array([0, 80, 315, 560, 736, 800, 900, 1000, 1280, 1350, 1480], dtype=float)
p_emp = np.array([8, 20, 620, 2200, 2056, 1800, 1500, 1250, 680, 320, 95], dtype=float)


def parse_variation(file_path):
    tag = file_path.stem.split("_")[-1]
    return (1 if tag[0] == "p" else -1) * float(tag[1:]) / 100.0


def compute_nrmse_range(t, y):
    y_emp = np.interp(t_emp, t, y)
    rmse = np.sqrt(np.mean((y_emp - p_emp) ** 2))
    return rmse / (p_emp.max() - p_emp.min())


def compute_slope(x, y):
    if len(x) < 2 or np.allclose(x, x[0]):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def extract_features(t, y):
    y = pd.Series(y).ffill().bfill().to_numpy(dtype=float)
    n = len(y)
    peak_idx = int(np.argmax(y))
    n20 = max(3, int(round(0.2 * n)))

    return [
        float(y[peak_idx]),
        float(t[peak_idx]),
        float(compute_slope(t[:n20], y[:n20])),
        float(np.trapezoid(y, t)),
        float(compute_slope(t[-n20:], y[-n20:])),
        float(y[-1]),
    ]


def get_modal_cluster(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    counts_dict = dict(zip(unique_labels.tolist(), counts.tolist()))

    non_noise_labels = unique_labels[unique_labels != -1]
    if len(non_noise_labels) == 0:
        return None, np.arange(len(labels)), counts_dict

    modal_label = max(non_noise_labels, key=lambda x: counts_dict[x])
    modal_idx = np.where(labels == modal_label)[0]

    return int(modal_label), modal_idx, counts_dict


results = []

for param in PARAM_NAMES:
    files = sorted(INPUT_DIR.glob(f"{CASE_NAME}_{param}_*.csv"), key=parse_variation)

    if not files:
        continue

    n_files = len(files)
    n_rows = math.ceil(n_files / N_COLS)

    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(5.2 * N_COLS, 4.0 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    cmap = plt.cm.cividis

    for ax, file_path in zip(axes, files):
        variation = parse_variation(file_path)

        df = pd.read_csv(file_path, usecols=["sim_id", "t", "P_ts_ABM"])
        matrix = df.pivot(index="sim_id", columns="t", values="P_ts_ABM").sort_index(axis=1)

        t = matrix.columns.to_numpy(dtype=float)
        sims = matrix.to_numpy(dtype=float)[:MAX_SIM]
        sims = pd.DataFrame(sims).ffill(axis=1).bfill(axis=1).to_numpy(dtype=float)

        X = np.asarray([extract_features(t, s) for s in sims], dtype=float)
        X = StandardScaler().fit_transform(X)

        labels = HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES,
            cluster_selection_method=CLUSTER_SELECTION_METHOD,
            allow_single_cluster=False
        ).fit_predict(X)

        modal_cluster, modal_idx, counts_dict = get_modal_cluster(labels)
        modal_mean = np.nanmean(sims[modal_idx], axis=0)
        nrmse = compute_nrmse_range(t, modal_mean)

        n_noise = int(np.sum(labels == -1))
        non_noise_labels = sorted(label for label in np.unique(labels) if label != -1)
        n_total = len(labels)

        results.append({
            "parameter": param,
            "variation": variation,
            "filename": file_path.name,
            "n_clusters_found": len(non_noise_labels),
            "n_noise_points": n_noise,
            "noise_share_pct": 100.0 * n_noise / n_total,
            "modal_cluster": "all_fallback" if modal_cluster is None else modal_cluster,
            "modal_cluster_size": len(modal_idx),
            "modal_cluster_share_pct": 100.0 * len(modal_idx) / n_total,
            "NRMSE_range_modal_cluster": nrmse,
        })

        color_map = {
            label: cmap(pos)
            for label, pos in zip(
                non_noise_labels,
                np.linspace(0.15, 0.9, max(len(non_noise_labels), 1))
            )
        }

        for i in range(sims.shape[0]):
            label = labels[i]
            ax.plot(
                t,
                sims[i],
                color="lightgray" if label == -1 else color_map[label],
                alpha=0.10,
                linewidth=0.8,
                zorder=1
            )

        ax.plot(
            t,
            modal_mean,
            color="lightblue",
            linestyle="--",
            linewidth=3,
            zorder=3
        )

        ax.scatter(
            t_emp,
            p_emp,
            color="salmon",
            s=100,
            zorder=10
        )

        pct = int(round(variation * 100))
        tag = f"p{pct}" if pct > 0 else f"m{abs(pct)}" if pct < 0 else "p0"

        ax.set_title(f"{tag} | NRMSE={nrmse:.3f}")
        ax.set_xlabel("t")
        ax.set_ylabel("Population")
        ax.set_ylim(bottom=0)

        legend_handles = []

        for label in non_noise_labels:
            share = 100.0 * counts_dict[label] / n_total
            legend_handles.append(
                plt.Line2D(
                    [0], [0],
                    color=color_map[label],
                    linewidth=2,
                    label=f"C{label}: {share:.1f}%"
                )
            )

        if n_noise > 0:
            legend_handles.append(
                plt.Line2D(
                    [0], [0],
                    color="lightgray",
                    linewidth=2,
                    label=f"Noise: {100.0 * n_noise / n_total:.1f}%"
                )
            )

        legend_handles.append(
            plt.Line2D(
                [0], [0],
                color="lightblue",
                linestyle="--",
                linewidth=3,
                label="Modal mean"
            )
        )

        legend_handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                color="salmon",
                linestyle="",
                markersize=8,
                label="Empirical data"
            )
        )

        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=False)

    for ax in axes[n_files:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if results:
    df_out = pd.DataFrame(results).sort_values(["parameter", "variation"])
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV saved to: {OUTPUT_CSV}")