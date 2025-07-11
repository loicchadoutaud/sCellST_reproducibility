from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sCellST_reproducibility.reproducibility_figures.utils_analyses import load_metrics
from scellst.constant import METRICS_DIR, DATA_DIR


def preprocess_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    list_hp = sorted(c for c in metrics.columns if c not in {"fold", "test_slide"} and not c.startswith("tag"))
    logger.info(f"Found {list_hp} parameters.")
    metrics.sort_values(by=list_hp, inplace=True)

    metrics["organ"] = metrics["genes"].str.split("_").str[0]
    metrics["genes"] = metrics["genes"].replace({
        "Prostate_50_hvg_bench": "50 HVG",
        "Prostate_200_hvg_bench": "200 HVG",
        "Prostate_500_hvg_bench": "500 HVG",
        "Prostate_1000_hvg_bench": "1000 HVG",
        "Prostate_2000_hvg_bench": "2000 HVG",
    })
    metrics["test_slide_fold"] = metrics["test_slide"].factorize(sort=True)[0]
    metrics = metrics[metrics["test_slide_fold"] == metrics["fold"].astype(int)]
    return metrics


def plot_gene_pcc_distribution(metrics: pd.DataFrame, save_path: Path, gene_order: list[str]):
    min_val, max_val, binwidth = -0.2, 0.8, 0.1
    fig, axes = plt.subplots(1, 3, figsize=(28, 6))

    sns.histplot(
        data=metrics, x="pcc", hue="genes", hue_order=gene_order,
        stat="proportion", multiple="dodge", common_norm=False, edgecolor="black",
        binwidth=binwidth, binrange=(min_val, max_val), palette="viridis",
        shrink=0.7, ax=axes[0]
    )
    axes[0].set_xticks(np.arange(min_val, max_val, binwidth))
    for b in np.arange(min_val, max_val, binwidth):
        axes[0].axvline(x=b, color="gray", linestyle="--")

    sns.kdeplot(
        data=metrics, x="pcc", hue="genes", hue_order=gene_order, palette="viridis",
        ax=axes[1], common_norm=False, fill=True, alpha=0.2
    )

    for ax in axes[:2]:
        ax.set_xlabel("pcc", size=20)
        ax.set_ylabel("Spot distribution", size=20)
        ax.set_title("Distribution of PCC for multiple numbers of training genes", fontsize=18)

    sns.barplot(
        data=metrics, x="genes", hue="genes", y="pcc",
        order=gene_order, hue_order=gene_order, palette="viridis", ax=axes[2]
    )
    axes[2].xaxis.label.set_size(18)
    axes[2].yaxis.label.set_size(18)
    axes[2].set_title("Mean PCC for multiple numbers of training genes", fontsize=18)

    sns.despine()
    fig.savefig(save_path / "gene_pcc_distribution.png", dpi=100, bbox_inches="tight")


def plot_top_50_hvg(metrics: pd.DataFrame, save_path: Path, gene_order: list[str]):
    top_50_hvg = metrics[metrics["genes"] == "50 HVG"]["gene"].unique()
    top_metrics = metrics[metrics["gene"].isin(top_50_hvg)]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=top_metrics, x="genes", y="pcc", hue="genes",
        palette="viridis", order=gene_order, hue_order=gene_order, ax=ax
    )
    ax.set_title("Mean PCC of top 50 HVG for different training gene sets.", fontsize=18)
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    sns.despine()
    fig.savefig(save_path / "gene_pcc_top_50_hvg.png", dpi=100, bbox_inches="tight")


def plot_rank_vs_metrics(metrics: pd.DataFrame, save_path: Path, rank_file: Path):
    top_2000 = metrics[metrics["genes"] == "2000 HVG"].groupby("gene")[["pcc", "scc"]].mean()
    gene_rank = pd.read_csv(rank_file, index_col=0)
    merged = top_2000.merge(gene_rank, left_on="gene", right_on="gene", how="left")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=merged, x="highly_variable_rank", y="pcc", ax=ax)
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    ax.set_title("Comparison between prediction performances and HVG ranks", fontsize=18)
    sns.despine()
    fig.savefig(save_path / "comparison_rank_metrics.png", dpi=100, bbox_inches="tight")


def run_gene_analysis():
    metrics_test = ["pcc", "scc"]
    exp_tag = "review-genes"
    gene_order = ["50 HVG", "200 HVG", "500 HVG", "1000 HVG", "2000 HVG"]
    save_path = Path("figures") / exp_tag
    save_path.mkdir(exist_ok=True, parents=True)

    metrics_dir = METRICS_DIR / exp_tag / "mil"
    metrics = load_metrics(metrics_dir, metrics_test)
    metrics = preprocess_metrics(metrics)

    plot_gene_pcc_distribution(metrics, save_path, gene_order)
    plot_top_50_hvg(metrics, save_path, gene_order)
    plot_rank_vs_metrics(metrics, save_path, DATA_DIR / "genes_Prostate_2000_hvg_bench.csv")


if __name__ == "__main__":
    sns.set_theme(style="white")
    run_gene_analysis()