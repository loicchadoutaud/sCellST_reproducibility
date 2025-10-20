from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sCellST_reproducibility.reproducibility_figures.utils_analyses import load_metrics
from sCellST_reproducibility.reproducibility_figures.utils_table import source_data
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


def plot_gene_pcc_distribution(metrics: pd.DataFrame, gene_order: list[str], save_path: Path, table_save_path: Path, ext: str = "svg") -> None:
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

    # Save source data
    src_df = source_data(
        df=metrics,
        x="genes",
        y="pcc",
    )
    src_df.to_csv(table_save_path / "pcc_hvg.csv")

    sns.despine()
    fig.savefig(save_path / f"gene_pcc_distribution.{ext}", dpi=100, bbox_inches="tight")


def plot_top_50_hvg(metrics: pd.DataFrame, gene_order: list[str], save_path: Path, table_save_path: Path, ext: str = "svg"):
    n_cats = metrics["genes"].nunique()
    common_genes = metrics.groupby("gene")["genes"].nunique().pipe(lambda s: s[s == n_cats]).index
    top_metrics = metrics[metrics["gene"].isin(common_genes)]

    # Figure
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=top_metrics, x="genes", y="pcc", hue="genes",
        palette="viridis", order=gene_order, hue_order=gene_order, ax=ax
    )
    ax.set_title("Mean PCC of top HVG for different training gene sets.", fontsize=18)
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    sns.despine()
    fig.savefig(save_path / f"gene_pcc_top_hvg.{ext}", dpi=100, bbox_inches="tight")

    # Save source data
    src_df = source_data(
        df=top_metrics,
        x="genes",
        y="pcc",
        hue="genes",
    )
    src_df.to_csv(table_save_path / "gene_pcc_top_hvg.csv")




def plot_rank_vs_metrics(metrics: pd.DataFrame, rank_file: Path, save_path: Path, table_save_path: Path, ext: str = "svg"):
    top_2000 = metrics[metrics["genes"] == "2000 HVG"].groupby("gene")[["pcc", "scc"]].mean()
    gene_rank = pd.read_csv(rank_file, index_col=0)
    merged = top_2000.merge(gene_rank, left_on="gene", right_on="gene", how="left")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=merged, x="highly_variable_rank", y="pcc", ax=ax)
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    ax.set_title("Comparison between prediction performances and HVG ranks", fontsize=18)

    # Save source data
    src_df = source_data(
        df=merged,
        x="highly_variable_rank",
        y="pcc",
    )
    src_df.to_csv(table_save_path / "comparison_rank_metrics.csv")

    sns.despine()
    fig.savefig(save_path / f"comparison_rank_metrics.{ext}", dpi=100, bbox_inches="tight")


def run_gene_analysis():
    metrics_test = ["pcc", "scc"]
    exp_tag = "review-genes"
    gene_order = ["50 HVG", "200 HVG", "500 HVG", "1000 HVG", "2000 HVG"]
    save_path = Path("figures") / exp_tag
    save_path.mkdir(exist_ok=True, parents=True)
    table_save_path = Path("tables") / exp_tag
    table_save_path.mkdir(exist_ok=True, parents=True)

    metrics_dir = METRICS_DIR / exp_tag / "mil"
    metrics = load_metrics(metrics_dir, metrics_test)
    metrics = preprocess_metrics(metrics)

    plot_gene_pcc_distribution(metrics, gene_order, save_path, table_save_path)
    plot_top_50_hvg(metrics, gene_order, save_path, table_save_path)
    plot_rank_vs_metrics(metrics, DATA_DIR / "genes_Prostate_2000_hvg_bench.csv", save_path, table_save_path, )


if __name__ == "__main__":
    sns.set_theme(style="white")
    run_gene_analysis()