import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from pathlib import Path
from loguru import logger

from sCellST_reproducibility.reproducibility_figures.utils_analyses import load_metrics
from scellst.constant import METRICS_DIR


def enrich_and_format(metrics: DataFrame) -> DataFrame:
    metrics["organ"] = metrics["genes"].str.split("_").str[0]
    if "model" in metrics.columns:
        metrics["model"] = metrics["model"].fillna("sCellST")
    metrics["genes"] = metrics["genes"].replace(
        {
            "Kidney_50_hvg_bench": "50 HVG",
            "Prostate_50_hvg_bench": "50 HVG",
        }
    )
    metrics["embedding_tag"] = metrics["embedding_tag"].replace(
        {
            "moco-Kidney-rn50_train": "MoCo",
            "moco-Prostate-rn50_train": "MoCo",
            "imagenet-rn50_train": "ImageNet",
        }
    )
    metrics.rename(columns={"embedding_tag": "embedding"}, inplace=True)
    return metrics


def plot_barplot(
    metrics: DataFrame, x: str, y: str, hue: str, title: str, save_path: Path
) -> None:
    # Select test slides
    df = metrics
    df["test_slide_fold"] = df.groupby("organ")["test_slide"].transform(
        lambda x: x.factorize(sort=True)[0]
    )
    df = df[(df["test_slide_fold"] == df["fold"].astype(int))]
    logger.info(df.groupby(hue)[y].mean())
    logger.info(df.groupby(hue).size())
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
    )
    ax.set_title(title)
    sns.move_legend(ax, "center left", bbox_to_anchor=(1.0, 0.5))
    sns.despine()
    fig.savefig(save_path, bbox_inches="tight")


def plot_benchmark(metrics: DataFrame) -> None:
    palette = sns.color_palette("magma", n_colors=5)
    palette = {model: color for model, color in zip(
        ["HisToGene", "THItoGene", "istar", "MclSTExp", "sCellST"], palette)}

    # Plot to inspect other metrics
    organ = "Prostate"
    genes = "50 HVG"

    # Select test slides
    metrics["test_slide_fold"] = metrics["test_slide"].factorize(sort=True)[0]
    df_plot = metrics[(metrics["test_slide_fold"] == metrics["fold"].astype(int))]

    x_order = ["HisToGene", "THItoGene", "MclSTExp", "sCellST"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    for metric, ax in zip(metrics_test, axes):
        sns.barplot(
            data=df_plot,
            x="model",
            y=metric,
            hue="model",
            order=x_order,
            palette=palette,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"{metric}", fontsize=20)
        ax.tick_params(axis='x', labelrotation=30)
    fig.suptitle(f"Metrics on multiple slide training {organ} with {genes}", y=1.02, fontsize=24, fontweight="bold")
    sns.despine()
    fig.savefig(str(save_path / f"{organ}-{genes}-metrics.png"), bbox_inches="tight")


if __name__ == "__main__":
    # Configuration
    list_models = ["istar", "HisToGene", "MclSTExp", "THItoGene"]
    save_path = Path("figures/review-benchmark")
    save_path.mkdir(parents=True, exist_ok=True)

    # Embeddings
    logger.info("Loading metrics...")
    metrics = load_metrics(
        list_metrics_dir=METRICS_DIR / "review-benchmark" / "mil",
        metrics_test=["pcc"],
    )
    logger.info("Formatting metrics...")
    metrics = enrich_and_format(metrics)
    logger.info("Plotting barplot...")
    plot_barplot(
        metrics,
        x="organ",
        y="pcc",
        hue="embedding",
        title="Comparison of PCC on multiple slide training.",
        save_path=save_path / "pcc_barplot.png",
    )
    logger.info("Done.")

    # # Metrics
    # metrics_test = ["pcc", "scc", "mae", "rmse"]
    # list_models = ["istar", "HisToGene", "MclSTExp", "THItoGene"]
    # list_metrics_dir = [METRICS_DIR / model for model in list_models] + [
    #     METRICS_DIR / "benchmark" / "mil",
    #     METRICS_DIR / "benchmark-multiple" / "mil",
    # ]
    #
    # logger.info("Loading metrics...")
    # metrics = load_metrics(
    #     list_metrics_dir=list_metrics_dir,
    #     metrics_test=metrics_test,
    #     pattern="fold=*;genes=Prostate_50_hvg_bench"
    #
    # )
    # logger.info("Formatting metrics...")
    # metrics = enrich_and_format(metrics)
    # logger.info("Plotting barplot...")
    # plot_benchmark(metrics)
    # logger.info("Done.")
