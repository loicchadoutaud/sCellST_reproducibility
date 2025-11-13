import scanpy as sc
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from pathlib import Path
from loguru import logger

from sCellST_reproducibility.reproducibility_figures.scalebar_ops import add_scale_bar
from sCellST_reproducibility.reproducibility_figures.utils_analyses import load_metrics, load_visium
from sCellST_reproducibility.reproducibility_figures.utils_plot import plot_he
from sCellST_reproducibility.reproducibility_figures.utils_table import source_data
from scellst.constant import METRICS_DIR
from sCellST_reproducibility.submit_scripts.script_constants import visium_slides


def enrich_and_format(metrics: DataFrame) -> DataFrame:
    logger.info("Formatting metrics...")
    metrics["multislide_training"] = metrics["train_slide"].isna()
    metrics["model"] = metrics["model"].fillna("sCellST")
    metrics["organ"] = metrics["genes"].str.split("_").str[0]

    gene_name_map = {
        "Kidney_50_hvg_bench": "50 HVG",
        "Kidney_50_svg_bench": "50 SVG",
        "Prostate_50_hvg_bench": "50 HVG",
        "Prostate_50_svg_bench": "50 SVG",
        "Kidney_500_hvg_bench": "500 HVG",
        "Kidney_500_svg_bench": "500 SVG",
        "Prostate_500_hvg_bench": "500 HVG",
        "Prostate_500_svg_bench": "500 SVG",
    }
    metrics["genes"] = metrics["genes"].replace(gene_name_map)
    return metrics


def plot_benchmark(metrics: DataFrame, save_path: Path, table_save_path: Path, ext: str = "svg") -> None:
    logger.info("Plotting benchmark figures...")
    palette = sns.color_palette("magma", n_colors=5)
    palette = {
        model: color
        for model, color in zip(
            ["HisToGene", "THItoGene", "istar", "MclSTExp", "sCellST"], palette
        )
    }
    x_order = ["50 HVG", "50 SVG", "500 HVG", "500 SVG"]

    for organ in metrics["organ"].unique():
        for metric in metrics_test:
            sns.set_style("whitegrid")
            fig, axes = plt.subplots(2, 1, figsize=(8, 14))

            # Single slide
            df_plot = metrics[
                (~metrics["multislide_training"])
                & (metrics["train_slide"] != metrics["test_slide"])
                & (metrics["organ"] == organ)
            ]
            sns.barplot(
                data=df_plot,
                x="genes",
                hue="model",
                y=metric,
                palette=palette,
                order=x_order,
                hue_order=palette.keys(),
                ax=axes[0],
            )
            axes[0].set_title("Single slide", fontsize=20)
            axes[0].set_xlabel("")
            axes[0].set_ylabel("")
            axes[0].tick_params(axis="x", labelsize=16)
            axes[0].tick_params(axis="y", labelsize=16)
            axes[0].set_ylim(0, None)
            sns.move_legend(
                axes[0],
                "center left",
                bbox_to_anchor=(1.05, -0.2),
                fontsize=20,
                title_fontsize=22,
            )

            # Save source data
            src_df = source_data(
                df=df_plot,
                x="genes",
                hue="model",
                y=metric,
            )
            src_df.to_csv(table_save_path / f"ss_{organ}_{metric}.csv")

            # Multi slide
            df_plot = metrics[
                (metrics["multislide_training"]) & (metrics["organ"] == organ)
            ].copy()
            df_plot["test_slide_fold"] = df_plot["test_slide"].factorize(sort=True)[0]
            df_plot = df_plot[df_plot["test_slide_fold"] == df_plot["fold"].astype(int)]
            hue_order = [
                m for m in palette if m != "istar"
            ]  # remove istar (no multi-slide)
            sns.barplot(
                data=df_plot,
                x="genes",
                hue="model",
                y=metric,
                palette=palette,
                order=x_order,
                hue_order=hue_order,
                ax=axes[1],
                legend=False,
            )
            axes[1].set_title("Leave one out", fontsize=20)
            axes[1].set_xlabel("")
            axes[1].set_ylabel("")
            axes[1].tick_params(axis="x", labelsize=16)
            axes[1].tick_params(axis="y", labelsize=16)
            axes[1].set_ylim(0, None)

            # Save source data
            src_df = source_data(
                df=df_plot,
                x="genes",
                hue="model",
                y=metric,
            )
            src_df.to_csv(table_save_path / f"ms_{organ}_{metric}.csv")

            sns.despine(fig)
            fig.suptitle(f"{organ} - {metric}", fontsize=24, fontweight="bold")
            fig.savefig(save_path / f"{organ}_{metric}.{ext}", bbox_inches="tight")
            plt.close(fig)


def generate_thumbnails(ext: str = "svg") -> None:
    logger.info("Generating thumbnails...")
    adata_dir = Path("/home/loic/Downloads")
    for organ, slides in visium_slides.items():
        images = []
        fig, axes = plt.subplots(4, 2, figsize=(8, 17), constrained_layout=True)
        for ax, slide in zip(axes.flatten(), slides):
            adata_path = adata_dir / f"{slide}.h5ad"
            adata = load_visium(adata_path)
            sc.pl.spatial(
                adata,
                img_key="downscaled_fullres",
                show=False,
                ax=ax,
            )
            img_key = next(iter(adata.uns["spatial"].keys()))
            downscale_factor = adata.uns["spatial"][img_key]["scalefactors"]["tissue_downscaled_fullres_scalef"]
            pixel_size = 55 / adata.uns["spatial"][img_key]["scalefactors"]["spot_diameter_fullres"]
            um_px = pixel_size / downscale_factor
            img_shape = adata.uns["spatial"][img_key]["images"]["downscaled_fullres"].shape
            add_scale_bar(ax, um_px, img_shape)
            ax.set_title(slide, fontsize=24)
            ax.axis("off")

        # Hide extra axes
        for ax in axes.flatten()[len(images) :]:
            ax.axis("off")
            ax.grid(False)

        fig.suptitle(f"{organ} dataset", fontsize=30, fontweight="bold", y=1.05)
        fig.savefig(save_path / f"{organ}_thumbnails.{ext}", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    # Configuration
    metrics_test = ["pcc", "scc"]
    list_models = ["istar", "HisToGene", "MclSTExp", "THItoGene"]
    list_metrics_dir = [METRICS_DIR / model for model in list_models] + [
        METRICS_DIR / "benchmark" / "mil",
        METRICS_DIR / "benchmark-multiple" / "mil",
    ]

    # Output path
    save_path = Path("figures/benchmark")
    table_save_path = Path("tables/benchmark")
    save_path.mkdir(parents=True, exist_ok=True)
    table_save_path.mkdir(parents=True, exist_ok=True)

    # Metric plots
    # metrics = load_metrics(list_metrics_dir, metrics_test=metrics_test)
    # metrics = enrich_and_format(metrics)
    # plot_benchmark(metrics, save_path, table_save_path)

    # Thumbnails
    generate_thumbnails()
    logger.info("Done.")
