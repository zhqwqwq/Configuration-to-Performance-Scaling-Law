import os
import json
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["STIXGeneral"]
plt.rcParams["mathtext.fontset"] = "stix"

MODEL_PATH = (
    "" # fill in the model path here. 
)
RESULTS_JSON_PATH = os.path.join(MODEL_PATH, "eval", "results.json")

OUT_DIR = "plots/scatter"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = {
    "optimizer": "marin",   
    "steplaw": "steplaw",
}
SPLITS = ["in_eval", "ood_eval"]  


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    rho = float(spearmanr(y_true, y_pred).correlation)
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = math.sqrt(mse)
    return {"spearman": rho, "mae": mae, "rmse": rmse}


def scatter_sorted(ax, x_gt, y_ncpl, y_chi):
    # sort by x for nicer visuals
    order_ncpl = np.argsort(x_gt)
    x1 = x_gt[order_ncpl]
    y1 = y_ncpl[order_ncpl]

    order_chi = np.argsort(x_gt)
    x2 = x_gt[order_chi]
    y2 = y_chi[order_chi]

    ax.plot(x1, y1, "o", markersize=5, alpha=0.7, color="#2C75B8", label="NCPL (Ours)")
    ax.plot(x2, y2, "o", markersize=5, alpha=0.8, color="#EB7373", label="Chinchilla")

    # y=x line + square axes
    all_vals = np.concatenate([x_gt, y_ncpl, y_chi]) if len(x_gt) else np.array([0.0, 1.0])
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    span = vmax - vmin if vmax > vmin else 1.0
    pad = 0.08 * span

    ax.plot([vmin, vmax], [vmin, vmax], "--", linewidth=1.2, color="#888888", label="y = x")
    ax.set_xlim(vmin - pad, vmax + pad)
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=18)


def add_spearman_box(ax, rho_ncpl, rho_chi, fontsize=18):
    if rho_ncpl is None or rho_chi is None:
        return
    txt = rf"$\rho_{{\mathrm{{NCPL}}}}={rho_ncpl:.4f}$" + "\n" + rf"$\rho_{{\mathrm{{Chinchilla}}}}={rho_chi:.4f}$"
    ax.text(
        0.03,
        0.97,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#888888", alpha=0.5),
    )


def load_results(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"results.json not found: {path}")
    with open(path, "r") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"results.json should be a dict, got {type(obj)}")
    return obj


def extract_xy(items: list, source_name: str):
    """Return arrays: x_gt, y_ncpl, y_chi for a given split list and dataset source."""
    x_gt, y_ncpl, y_chi = [], [], []

    for it in items:
        if it.get("meta_data", {}).get("source") != source_name:
            continue

        try:
            chin = float(it["chinchilla"])
            label = float(it["label_seqs"]["train_loss"][0])
            gen = float(it["generated_seqs"]["train_loss"][0])
        except Exception:
            continue

        x = label + chin
        x_gt.append(x)
        y_ncpl.append(gen + chin)
        y_chi.append(chin)

    return (
        np.array(x_gt, dtype=float),
        np.array(y_ncpl, dtype=float),
        np.array(y_chi, dtype=float),
    )


def plot_two_panel(all_results: dict, dataset_key: str, pdf_name: str):
    source_name = DATASETS[dataset_key]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=False, sharey=False)

    panel_splits = [("in_eval", "ID"), ("ood_eval", "OOD")]

    for i, (split, title) in enumerate(panel_splits):
        ax = axes[i]
        items = all_results.get(split, [])
        if not isinstance(items, list) or len(items) == 0:
            ax.set_visible(False)
            continue

        x_gt, y_ncpl, y_chi = extract_xy(items, source_name)
        if len(x_gt) == 0:
            ax.set_visible(False)
            continue

        m_ncpl = compute_metrics(x_gt, y_ncpl) if len(x_gt) > 1 else {"spearman": None}
        m_chi = compute_metrics(x_gt, y_chi) if len(x_gt) > 1 else {"spearman": None}

        scatter_sorted(ax, x_gt, y_ncpl, y_chi)
        add_spearman_box(ax, m_ncpl["spearman"], m_chi["spearman"], fontsize=20)

        ax.set_xlabel("Ground-truth loss", fontsize=22)
        ax.set_ylabel("Predicted loss" if i == 0 else "", fontsize=22)
        ax.set_title(title, fontsize=24, pad=10)

    handles = [
        Line2D([], [], linestyle="--", color="#888888", linewidth=1.5, label="y = x"),
        Line2D([], [], marker="o", linestyle="", color="#2C75B8", markersize=8, label="NCPL (Ours)"),
        Line2D([], [], marker="o", linestyle="", color="#EB7373", markersize=8, label="Chinchilla"),
    ]
    axes[0].legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(1.032, -0.032),
        ncol=1,
        fontsize=16,
        frameon=True,
        labelspacing=0.1,
        handletextpad=0.3,
        columnspacing=0.4,
        borderpad=0.2,
    )

    fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.28, wspace=-0.25)

    out_path = os.path.join(OUT_DIR, pdf_name)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_path}")


def main():
    all_results = load_results(RESULTS_JSON_PATH)
    plot_two_panel(all_results, "optimizer", "scatter_marin.pdf")
    plot_two_panel(all_results, "steplaw", "scatter_steplaw.pdf")


if __name__ == "__main__":
    main()
