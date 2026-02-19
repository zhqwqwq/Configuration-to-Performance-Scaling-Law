from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd


# ===================== paths =====================

DEFAULT_RESULTS_JSON = Path("") # Fill in model path here, e.g. "NCPL-intermediate/eval/results.json"

PLOTS_DIR = Path(__file__).resolve().parent

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["STIXGeneral"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update(
    {
        "font.size": 28,
        "axes.titlesize": 28,
        "axes.labelsize": 28,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "legend.fontsize": 24,
    }
)

GT_MARKER = "o"
PRED_MARKER = "^"
GT_LS = "-"
PRED_LS = "--"

COLORS = ["#EB7373", "#8C6BB1", "#E8A35C", "#3C9D9B", "#7FB800", "#FF7F00", "#6A5ACD"]

TOKEN_RE = re.compile(r"-(\d+(?:\.\d+)?)B-")


def infer_max_tokens_b(run_name: str) -> float:
    match = TOKEN_RE.search(run_name)
    if not match:
        raise ValueError(f"Unable to infer token budget from name: {run_name}")
    return float(match.group(1))


SplitName = Literal["train_eval", "in_eval", "ood_eval"]


def _extract_first(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, list) and len(x) > 0:
        v = x[0]
        return float(v) if v is not None else None
    return None


def load_results(results_path: Path) -> dict[str, list[dict[str, Any]]]:
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found: {results_path}")
    with results_path.open("r") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("results.json must be a dict with keys: train_eval/in_eval/ood_eval.")
    return obj  # type: ignore[return-value]


def build_split_df(
    results: dict[str, list[dict[str, Any]]],
    split: SplitName,
    metric_key: str = "intermediate_train_loss",
    pred_source: Literal["generated_tf_seqs", "generated_seqs"] = "generated_tf_seqs",
) -> pd.DataFrame:
    """
    Convert results[split] list into a DataFrame compatible with plotting code.

    Output columns:
      - name
      - ratio             (mapped from 'frac')
      - true_train_loss   (from label_seqs[metric_key][0])
      - pred_train_loss   (from pred_source[metric_key][0])
    """
    rows: list[dict[str, Any]] = []
    items = results.get(split, [])
    if not isinstance(items, list):
        raise ValueError(f"results[{split}] must be a list, got {type(items)}")

    for it in items:
        if not isinstance(it, dict):
            continue

        name = it.get("name", None)
        frac = it.get("frac", None)

        label_seqs = it.get("label_seqs", {}) or {}
        pred_seqs = it.get(pred_source, {}) or {}

        true_v = _extract_first(label_seqs.get(metric_key))
        pred_v = _extract_first(pred_seqs.get(metric_key))
        if true_v == None: # final loss prediction
            continue
        chinchilla = it["chinchilla"]
        rows.append(
            {
                "name": name,
                "ratio": float(frac) if frac is not None else None,
                "true_train_loss": true_v + chinchilla,
                "pred_train_loss": pred_v + chinchilla,
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["name", "ratio"])
    df["ratio"] = df["ratio"].astype(float)

    return df


def plot_multiple_runs(
    ax: plt.Axes,
    names_list: list[tuple[str, str]],
    data: pd.DataFrame,
    title: str | None = None,
    legend_fontsize: float = 20.5,
    legend_loc: str = "upper right",
) -> None:
    used_opts: list[str] = []
    max_tokens_b = 0.0

    for opt_name, run_name in names_list:
        subset = data[data["name"] == run_name].copy()
        if subset.empty:
            print(f"[warn] No entries found for name: {run_name}")
            continue

        subset = subset.sort_values("ratio")
        subset = subset.dropna(subset=["true_train_loss", "pred_train_loss"])

        if subset.empty:
            print(f"[warn] All entries missing losses for name: {run_name}")
            continue

        tokens_b = subset["ratio"] * infer_max_tokens_b(run_name)
        max_tokens_b = max(max_tokens_b, float(tokens_b.max()))
        col = COLORS[len(used_opts) % len(COLORS)]
        used_opts.append(opt_name)

        # Ground truth
        ax.plot(
            tokens_b,
            subset["true_train_loss"],
            linestyle=GT_LS,
            marker=GT_MARKER,
            color=col,
            linewidth=2.5,
            markersize=8,
        )

        # Prediction
        ax.plot(
            tokens_b,
            subset["pred_train_loss"],
            linestyle=PRED_LS,
            marker=PRED_MARKER,
            color=col,
            linewidth=2.5,
            markersize=8,
        )

    # labels / axes
    ax.set_xlabel("Training data (B tokens)")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Keep your original y-range; adjust if needed
    ax.set_ylim(3.08, 4)

    if max_tokens_b > 0:
        ax.set_xlim(0, max_tokens_b)
    if title:
        ax.set_title(title)

    # ====== legends: (1) color legend for optimizer ======
    seen: set[str] = set()
    ordered_opts = [o for o in used_opts if not (o in seen or seen.add(o))]

    color_handles = [
        mlines.Line2D(
            [],
            [],
            linestyle="-",
            color=COLORS[i % len(COLORS)],
            linewidth=4,
            label=o,
        )
        for i, o in enumerate(ordered_opts)
    ]

    ax.legend(
        handles=color_handles,
        loc=legend_loc,
        frameon=True,
        fontsize=legend_fontsize,
        handlelength=1.2,
    )


def add_style_legend(fig: plt.Figure, ax: plt.Axes | None = None) -> None:
    style_handles = [
        mlines.Line2D(
            [],
            [],
            linestyle=PRED_LS,
            marker=PRED_MARKER,
            markersize=10,
            markerfacecolor="black",
            markeredgecolor="black",
            color="black",
            linewidth=2.5,
            label="NCPL(Ours)",
        ),
        mlines.Line2D(
            [],
            [],
            linestyle=GT_LS,
            marker=GT_MARKER,
            markersize=10,
            markerfacecolor="black",
            markeredgecolor="black",
            color="black",
            linewidth=2.5,
            label="Ground-truth loss",
        ),
    ]

    if ax is None:
        fig.legend(
            handles=style_handles,
            loc="lower center",
            ncol=2,
            frameon=True,
            bbox_to_anchor=(0.5, -0.5),
            handlelength=2,
        )
        return

    ax.legend(
        handles=style_handles,
        loc="center",
        ncol=2,
        frameon=True,
        handlelength=2,
        bbox_to_anchor=(0.5, -0.5),
    )
    ax.set_axis_off()


def main(results_json: Path = DEFAULT_RESULTS_JSON) -> None:
    results = load_results(results_json)

    # Build split dfs
    in_df = build_split_df(results, "in_eval", metric_key="intermediate_train_loss", pred_source="generated_tf_seqs")
    ood_df = build_split_df(results, "ood_eval", metric_key="intermediate_train_loss", pred_source="generated_tf_seqs")

    # ===================== your selections =====================
    names_list_in_left = [
        ("kron", "sweep-300m-12B-kronb21812lr0.001-wd0.7-b10.95-plr0.2-pis1-gn1-no-48c292"),
        ("scion", "sweep-300m-12B-scion3f6fc4lr0.008-wd0.1-minlr0-warmup0-b10.95-gn-7c6102"),
        ("nadam", "sweep-300m-12B-nadamw59ffb0lr0.004-wd0.1-minlr0-warmup2000-b10.9-f2e4bd"),
    ]

    names_list_in_right = [
        ("lr8e-3wd0.1bs256", "sweep-130m-21B-adamw9248c1lr0.008-wd0.1-minlr0-warmup1000-b10.9--7ae1a8"),
        ("lr4e-3wd0.1bs256", "sweep-130m-21B-adamwdb2f39lr0.004-wd0.1-minlr0-warmup1000-b10.9--cf39b6"),
        ("lr8e-3wd0  bs256", "sweep-130m-21B-adamw549010lr0.008-wd0-minlr0-warmup1000-b10.9-b2-c08f03"),
        ("lr8e-3wd0.1bs128", "sweep-130m-21B-adamw292ad2lr0.008-wd0.1-minlr0-warmup1000-b10.9--b88aa1"),
    ]

    names_list_ood_left = [
        ("lion", "sweep-520m-10B-lionaf4d5dlr0.001-wd0.4-minlr0-warmup2000-b10.9-b-d38392"),
        ("mars", "sweep-520m-10B-marsfbbfc9lr0.008-wd0.1-minlr0-warmup2000-b10.95--e8c9e5"),
        ("muon", "sweep-520m-10B-muon9e8901lr0.008-wd0-minlr0-warmup0-b10.8-b20.98-f12151"),
    ]
    names_list_ood_right = [
        ("wd 0", "sweep-520m-10B-muon9e8901lr0.008-wd0-minlr0-warmup0-b10.8-b20.98-f12151"),
        ("wd 0.1", "sweep-520m-10B-muon076a83lr0.008-wd0.1-minlr0-warmup0-b10.8-b20.-d5cd2a"),
        ("wd 0.2", "sweep-520m-10B-muon3e335flr0.008-wd0.2-minlr0-warmup0-b10.8-b20.-b2ac88"),
    ]

    # ===================== Figure 1: optimizer (ID vs OOD) =====================
    fig_opt = plt.figure(figsize=(11, 7))
    gs_opt = fig_opt.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.0, 0.2],
        hspace=0.18,
        wspace=0.19,
    )
    axes_opt = [fig_opt.add_subplot(gs_opt[0, 0]), fig_opt.add_subplot(gs_opt[0, 1])]
    ax_opt_legend = fig_opt.add_subplot(gs_opt[1, :])

    plot_multiple_runs(axes_opt[0], names_list_in_left, in_df, legend_fontsize=26)
    plot_multiple_runs(axes_opt[1], names_list_ood_left, ood_df, legend_fontsize=26)

    axes_opt[0].set_title("ID", fontweight="normal")
    axes_opt[1].set_title("OOD", fontweight="normal")
    axes_opt[1].set_ylabel("")

    fig_opt.subplots_adjust(left=0.13, right=0.99, bottom=0.15, top=0.90)
    add_style_legend(fig_opt, ax_opt_legend)

    output_path_opt = PLOTS_DIR / "intermediate" / "intermediate_optimizer_arxiv.pdf"
    os.makedirs(output_path_opt.parent, exist_ok=True)
    fig_opt.savefig(output_path_opt)
    print(f"[ok] wrote {output_path_opt}")

    # ===================== Figure 2: hyperparameters (ID vs OOD) =====================
    fig_hp = plt.figure(figsize=(11, 7))
    gs_hp = fig_hp.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.0, 0.2],
        hspace=0.18,
        wspace=0.19,
    )
    axes_hp = [fig_hp.add_subplot(gs_hp[0, 0]), fig_hp.add_subplot(gs_hp[0, 1])]
    ax_hp_legend = fig_hp.add_subplot(gs_hp[1, :])

    plot_multiple_runs(
        axes_hp[0],
        names_list_in_right,
        in_df,
        legend_fontsize=16,
        legend_loc="upper right",
    )
    plot_multiple_runs(axes_hp[1], names_list_ood_right, ood_df, legend_fontsize=16)

    axes_hp[0].set_title("ID", fontweight="normal")
    axes_hp[1].set_title("OOD", fontweight="normal")
    axes_hp[1].set_ylabel("")

    fig_hp.subplots_adjust(left=0.13, right=0.99, bottom=0.15, top=0.90)
    add_style_legend(fig_hp, ax_hp_legend)

    output_path_hp = PLOTS_DIR / "intermediate" / "intermediate_hyperparameters_arxiv.pdf"
    os.makedirs(output_path_hp.parent, exist_ok=True)
    fig_hp.savefig(output_path_hp)
    print(f"[ok] wrote {output_path_hp}")

    plt.show()


if __name__ == "__main__":
    main()
