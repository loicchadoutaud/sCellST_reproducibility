from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
from loguru import logger

from scellst.constant import PREDS_DIR, METRICS_DIR
from scellst.dataset.data_handler import XeniumHandler
from scellst.plots.plot_spatial import plot_top_genes
from sCellST_reproducibility.reproducibility_figures.utils_analyses import (
    load_predictions, load_metrics,
)
from sCellST_reproducibility.reproducibility_figures.utils_plot import plot_he


def rotate_image(adata: AnnData) -> AnnData:
    adata.uns["spatial"]["ST"]["images"]["downscaled_fullres"] = np.swapaxes(
        adata.uns["spatial"]["ST"]["images"]["downscaled_fullres"], 0, 1
    )
    adata.obsm["spatial"] = adata.obsm["spatial"][:, [1, 0]]
    return adata


def crop_image(
    adata: AnnData, x_min: int, x_max: int, y_min: int, y_max: int
) -> AnnData:
    img = adata.uns["spatial"]["ST"]["images"]["downscaled_fullres"]
    scale_factor = adata.uns["spatial"]["ST"]["scalefactors"][
        "tissue_downscaled_fullres_scalef"
    ]

    y_min_ = int(y_min * scale_factor)
    y_max_ = int(y_max * scale_factor)
    x_min_ = int(x_min * scale_factor)
    x_max_ = int(x_max * scale_factor)

    adata.uns["spatial"]["ST"]["images"]["downscaled_fullres"] = img[
        y_min_:y_max_, x_min_:x_max_, :
    ]
    adata.obsm["spatial"][:, 0] -= x_min
    adata.obsm["spatial"][:, 1] -= y_min
    return adata


def plot_xenium_exp(
    data_dir: Path,
    slide_id: str,
    list_genes: list[str],
    shape_name: str,
    save_path: Path,
) -> AnnData:

    slide_dir = save_path / slide_id
    slide_dir.mkdir(parents=True, exist_ok=True)

    components = [
        "embedding_tag=moco-TENX39-rn50_self;genes=SVG:1000;scale=global_scaling;train_slide=TENX39",
        f"test_slide={slide_id}",
        "infer_mode=inference.h5ad",
    ]
    prediction_filename = ";".join(components)
    prediction_path = PREDS_DIR / "xenium" / "xenium" / prediction_filename
    cell_adata_pred = load_predictions(prediction_path)
    cell_adata = XeniumHandler().load_and_preprocess_data(
        data_dir,
        slide_id,
        filter_genes=True,
        filter_cells=True,
        normalize=False,
        log1p=True,
        embedding_path=None,
        shape_name=shape_name,
    )
    cell_adata.X = cell_adata.X.toarray()

    obs_names = list(set(cell_adata.obs_names) & set(cell_adata_pred.obs_names))
    cell_adata = cell_adata[obs_names]
    cell_adata_pred = cell_adata_pred[obs_names]

    if slide_id == "TENX95":
        rotate_image(cell_adata)
        rotate_image(cell_adata_pred)
        cell_adata = crop_image(cell_adata, 2500, 53738, 400, 47000)
        cell_adata_pred = crop_image(cell_adata_pred, 2500, 53738, 400, 47000)

    plot_he(
        cell_adata, title=f"{slide_id} H&E", save_path=slide_dir / f"he_{slide_id}.png"
    )
    plot_top_genes(cell_adata, cell_adata_pred, list_genes, slide_dir / f"gene_exp.png")

    return cell_adata


def plot_metrics(
    metrics: pd.DataFrame, save_path: Path, metrics_test: list[str]
):
    for metric in metrics_test:
        logger.info(metrics.groupby("test slide")[metric].median())
        logger.info(metrics.groupby("test slide").size())

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.violinplot(
            data=metrics, x="test slide", y=metric, hue="test slide", palette="magma"
        )
        ax.set_title(f"SVG {metric}")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=2)
        sns.despine()
        fig.savefig(
            save_path / f"{metric}_distribution.png", bbox_inches="tight", dpi=150
        )
        plt.close(fig)


def save_metrics(
        metrics: pd.DataFrame, save_path: Path,
) -> None:
    df = metrics.pivot(index="gene", columns="test slide", values="pcc")
    df["mean"] = df.mean(axis=1)
    df = df.round(3)
    df.sort_values(by="mean", ascending=False, inplace=True)
    df.to_csv(save_path / "all_metrics.csv")



def run_xenium_analysis():
    data_dir = Path("../../hest_data")
    save_path = Path("figures/xenium")
    save_path.mkdir(parents=True, exist_ok=True)
    metrics_test = ["pcc", "scc"]

    # Single slide plots
    shape_name = "xenium_nucleus"
    xenium_slides = {
        "NCBI785": ["KRT8", "PTPRC"],
        "TENX95": ["EPCAM", "CD3E"],
    }
    for slide_id, genes in xenium_slides.items():
        plot_xenium_exp(data_dir, slide_id, genes, shape_name, save_path)

    # Multiple slide experiments
    metrics_dir = METRICS_DIR / "xenium" / "xenium"
    metrics_df = load_metrics(metrics_dir, metrics_test)
    metrics_df = metrics_df.rename(columns={"test_slide": "test slide"})
    logger.info(metrics_df[["pcc", "scc"]].mean())
    plot_metrics(metrics_df, save_path, metrics_test)
    save_metrics(metrics_df, save_path)


if __name__ == "__main__":
    run_xenium_analysis()
